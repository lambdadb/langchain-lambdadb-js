import { VectorStore } from "@langchain/core/vectorstores";
import { Document } from "@langchain/core/documents";
import { EmbeddingsInterface } from "@langchain/core/embeddings";
import { LambdaDBClient } from "@functional-systems/lambdadb";

import {
  LambdaDBConfig,
  CreateCollectionOptions,
  DocumentFilter,
  DeleteOptions,
  MaxMarginalRelevanceSearchOptions,
  CollectionInfo,
  RetryOptions,
} from "./types.js";
import {
  lambdaDBToDocument,
  validateConfig,
  validateVectorDimensions,
  handleLambdaDBError,
  generateDocumentId,
  batchArray,
  withRetry,
  DEFAULT_RETRY_OPTIONS,
} from "./utils.js";

/**
 * LambdaDB vector store implementation for LangChain
 */
export class LambdaDBVectorStore extends VectorStore {
  declare FilterType: DocumentFilter;

  private client: LambdaDBClient;
  private collection: ReturnType<LambdaDBClient["collection"]>;
  private config: LambdaDBConfig;
  private textField: string;
  private vectorField: string;
  private retryOptions: RetryOptions;

  constructor(embeddings: EmbeddingsInterface, config: LambdaDBConfig) {
    super(embeddings, config);
    
    validateConfig(config);
    
    // Set configuration with defaults
    this.config = {
      textField: "content",
      vectorField: "vector", // Use 'vector' to match LambdaDB conventions
      validateCollection: false,
      defaultConsistentRead: true, // Use consistent reads by default for immediate consistency
      ...config,
    };
    
    this.textField = this.config.textField!;
    this.vectorField = this.config.vectorField!;
    this.retryOptions = { ...DEFAULT_RETRY_OPTIONS, ...(config.retryOptions || {}) };
    
    // Initialize LambdaDB client (0.3.x SDK: LambdaDBClient with collection-scoped API)
    this.client = new LambdaDBClient({
      projectApiKey: config.projectApiKey,
      ...(config.serverURL && { serverURL: config.serverURL }),
      timeoutMs: 30000, // 30 second timeout for all operations
    });
    this.collection = this.client.collection(this.config.collectionName);
    
    // Validate collection exists if requested
    if (this.config.validateCollection) {
      this.validateCollectionExists().catch((error) => {
        throw new Error(`Collection validation failed: ${error.message}`);
      });
    }
  }

  /**
   * Return the vector store type identifier
   */
  _vectorstoreType(): string {
    return "lambdadb";
  }

  /**
   * Add documents to the vector store
   */
  async addDocuments(documents: Document[]): Promise<void> {
    try {
      // Handle empty document array
      if (documents.length === 0) {
        return;
      }

      const texts = documents.map(({ pageContent }) => pageContent);
      const embeddings = await this.embeddings.embedDocuments(texts);
      
      await this.addVectors(embeddings, documents);
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Add vectors with associated documents to the store
   */
  async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
    try {
      // Validate input lengths match
      if (vectors.length !== documents.length) {
        throw new Error("Vectors and documents length mismatch");
      }

      // Validate vector dimensions
      if (vectors.length > 0) {
        validateVectorDimensions(vectors[0], this.config.vectorDimensions);
      }

      // Ensure collection exists
      await this.ensureCollectionExists();

      // Convert documents to LambdaDB format using configurable field names
      const lambdaDBDocs = vectors.map((vector, idx) => {
        const doc = documents[idx];
        const docData: Record<string, any> = {
          id: generateDocumentId(), // Use regular id field  
          [this.textField]: doc.pageContent,
          [this.vectorField]: vector,
          ...doc.metadata,
        };
        return docData;
      });

      // Batch upsert documents using correct API structure
      const batchSize = 100; // Adjust based on LambdaDB limits
      const batches = batchArray(lambdaDBDocs, batchSize);

      for (const batch of batches) {
        await withRetry(async () => {
          await this.collection.docs.upsert({ docs: batch });
        }, this.retryOptions);
      }
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Perform similarity search with scores
   */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: DocumentFilter
  ): Promise<[Document, number][]> {
    try {
      validateVectorDimensions(query, this.config.vectorDimensions);

      // Query LambdaDB for similar vectors using correct KNN API structure with retry
      const response = await withRetry(async () => {
        return await this.collection.query({
          size: k,
          query: {
            knn: {
              field: this.vectorField,
              queryVector: query,
              k: k
            }
          },
          consistentRead: this.config.defaultConsistentRead,
        });
      }, this.retryOptions);

      // Convert results to LangChain format
      const formattedResults: [Document, number][] = response.docs.map((result) => {
        const doc = lambdaDBToDocument(result.doc, this.textField);
        const score = result.score || 0;
        return [doc, score];
      });

      // Apply client-side filtering if needed
      if (filter) {
        return formattedResults.filter(([doc]) => filter(doc));
      }

      return formattedResults;
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Perform similarity search without scores
   */
  async similaritySearch(
    query: string,
    k = 4,
    filter?: DocumentFilter
  ): Promise<Document[]> {
    const embeddings = await this.embeddings.embedQuery(query);
    const results = await this.similaritySearchVectorWithScore(embeddings, k, filter);
    return results.map(([doc]) => doc);
  }

  /**
   * Create a new collection with vector index
   */
  async createCollection(options?: Partial<CreateCollectionOptions>): Promise<void> {
    try {
      // Create collection with proper index configuration (0.3.x: createCollection takes request body)
      await withRetry(async () => {
        await this.client.createCollection({
          collectionName: this.config.collectionName,
          indexConfigs: {
            [this.vectorField]: {
              type: "vector" as const,
              dimensions: this.config.vectorDimensions,
              similarity: (this.config.similarityMetric?.toLowerCase() || "cosine") as "cosine" | "euclidean" | "dot_product" | "max_inner_product",
            },
            ...(this.config.indexConfig || {}),
            ...(options?.indexConfig || {}),
          },
        });
      }, this.retryOptions);

      // Wait for collection to become ACTIVE before proceeding
      await this.waitForCollectionActive();
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Wait for collection to become ACTIVE
   */
  private async waitForCollectionActive(maxWaitTimeMs: number = 30000): Promise<void> {
    const startTime = Date.now();
    const pollInterval = 1000; // Check every 1 second

    while (Date.now() - startTime < maxWaitTimeMs) {
      try {
        const info = await this.getCollectionInfo();
        
        if (info.status === 'ACTIVE') {
          return; // Collection is ready
        }
        
        if (info.status === 'FAILED' || info.status === 'ERROR') {
          throw new Error(`Collection creation failed with status: ${info.status}`);
        }

        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, pollInterval));
        
      } catch (error) {
        // If we can't get collection info, it might still be creating
        if (Date.now() - startTime < maxWaitTimeMs) {
          await new Promise(resolve => setTimeout(resolve, pollInterval));
          continue;
        }
        throw error;
      }
    }

    throw new Error(`Collection did not become ACTIVE within ${maxWaitTimeMs}ms`);
  }

  /**
   * Delete the collection
   */
  async deleteCollection(): Promise<void> {
    try {
      await this.collection.delete();
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Delete documents from the vector store (LangChain VectorStore interface).
   * Maps _params to DeleteOptions and delegates to deleteDocuments().
   * Requires explicit params to avoid accidental full collection wipe.
   *
   * @param _params - One of: { ids?: string[] } | { filter?: DocumentFilter } | { deleteAll: true }.
   *                 Omitted or empty → throws (no default to deleteAll).
   */
  async delete(_params?: Record<string, any>): Promise<void> {
    if (!_params || Object.keys(_params).length === 0) {
      throw new Error(
        "delete() requires explicit params to avoid accidental wipe. Pass one of: { ids: string[] }, { filter: (doc) => boolean }, or { deleteAll: true }"
      );
    }
    if (_params.deleteAll === true) {
      await this.deleteDocuments({ deleteAll: true });
      return;
    }
    if (Array.isArray(_params.ids) && _params.ids.length > 0) {
      await this.deleteDocuments({ ids: _params.ids });
      return;
    }
    if (
      typeof _params.filter === "function" ||
      (typeof _params.filter === "object" && _params.filter !== null) ||
      typeof _params.filter === "string"
    ) {
      await this.deleteDocuments({ filter: _params.filter });
      return;
    }
    throw new Error(
      "delete() requires one of: ids (string[]), filter (LambdaDB object/query string or function), or deleteAll (true)"
    );
  }

  /**
   * Delete documents from the vector store
   */
  async deleteDocuments(options: DeleteOptions): Promise<void> {
    try {
      if (options.deleteAll) {
        // Delete all documents using LambdaDB filter with wildcard match-all.
        // See https://docs.lambdadb.ai/guides/documents/delete-data
        await this.collection.docs.delete({
          filter: { queryString: { query: "*:*" } },
        });
      } else if (options.ids && options.ids.length > 0) {
        // Delete documents by IDs
        await this.collection.docs.delete({ ids: options.ids });
      } else if (options.filter !== undefined && options.filter !== null) {
        if (typeof options.filter === "function") {
          // Client-side filter: fetch all, filter, delete by ids (less efficient for large collections)
          const allDocs = await this.getAllDocuments();
          const docsToDelete = allDocs.filter(options.filter as DocumentFilter);
          const idsToDelete = docsToDelete
            .map((doc) => doc.metadata.id)
            .filter((id) => id);
          if (idsToDelete.length > 0) {
            await this.deleteDocuments({ ids: idsToDelete });
          }
        } else {
          // LambdaDB filter (object or query string): pass to API for server-side delete
          const apiFilter =
            typeof options.filter === "string"
              ? { queryString: { query: options.filter } }
              : (options.filter as Record<string, unknown>);
          await this.collection.docs.delete({ filter: apiFilter });
        }
      } else {
        throw new Error("Must provide either ids, filter, or deleteAll option");
      }
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Maximum marginal relevance search
   */
  async maxMarginalRelevanceSearch(
    query: string,
    options: MaxMarginalRelevanceSearchOptions,
    _callbacks?: any
  ): Promise<Document[]> {
    const {
      k = 4,
      fetchK = 20,
      lambda = 0.5,
      filter,
    } = options;

    try {
      // Convert filter to function if needed
      const filterFn: DocumentFilter | undefined = typeof filter === 'function' 
        ? filter as DocumentFilter 
        : undefined;

      // First, get more candidates than needed
      const candidateResults = await this.similaritySearchVectorWithScore(
        await this.embeddings.embedQuery(query),
        fetchK,
        filterFn
      );

      if (candidateResults.length === 0) {
        return [];
      }

      // Extract embeddings for MMR calculation (this would require storing vectors)
      // For now, we'll implement a simplified version that just returns top-k results
      // A full MMR implementation would require vector storage and access
      const selected: Document[] = [];
      const candidates = candidateResults.map(([doc]) => doc);

      // Select first document (highest similarity)
      if (candidates.length > 0) {
        selected.push(candidates[0]);
      }

      // For remaining selections, balance relevance and diversity
      // This is a simplified MMR - a full implementation would calculate
      // vector similarities between candidates
      while (selected.length < k && selected.length < candidates.length) {
        let bestIdx = -1;
        let bestScore = -Infinity;

        for (let i = 0; i < candidates.length; i++) {
          const candidate = candidates[i];
          if (selected.includes(candidate)) continue;

          // Simplified scoring: favor later results (more diverse)
          // In full MMR, this would be: lambda * similarity - (1-lambda) * max_similarity_to_selected
          const diversityBonus = (1 - lambda) * (i / candidates.length);
          const relevanceScore = lambda * (1 - i / candidates.length);
          const score = relevanceScore + diversityBonus;

          if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
          }
        }

        if (bestIdx >= 0) {
          selected.push(candidates[bestIdx]);
        } else {
          break;
        }
      }

      return selected.slice(0, k);
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Get collection information
   */
  async getCollectionInfo(): Promise<CollectionInfo> {
    try {
      const response = await this.collection.get();
      const col = response.collection;

      return {
        name: col.collectionName ?? this.config.collectionName,
        status: col.collectionStatus ?? "unknown",
        documentCount: col.numDocs,
        indexConfigs: col.indexConfigs,
        createdAt: undefined,
        updatedAt: undefined,
      };
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Static method to create LambdaDBVectorStore from texts
   */
  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: EmbeddingsInterface,
    config: LambdaDBConfig
  ): Promise<LambdaDBVectorStore> {
    const docs = texts.map((text, idx) => {
      const metadata = Array.isArray(metadatas) ? metadatas[idx] || {} : metadatas || {};
      return new Document({ pageContent: text, metadata });
    });

    return LambdaDBVectorStore.fromDocuments(docs, embeddings, config);
  }

  /**
   * Static method to create LambdaDBVectorStore from documents
   */
  static async fromDocuments(
    docs: Document[],
    embeddings: EmbeddingsInterface,
    config: LambdaDBConfig
  ): Promise<LambdaDBVectorStore> {
    const instance = new LambdaDBVectorStore(embeddings, config);
    await instance.addDocuments(docs);
    return instance;
  }

  /**
   * Validate that the collection exists
   */
  private async validateCollectionExists(): Promise<void> {
    try {
      await this.collection.get();
    } catch (error) {
      throw new Error(`Collection '${this.config.collectionName}' does not exist: ${error}`);
    }
  }

  /**
   * Get all documents from the collection (for internal operations, e.g. filter-based delete).
   * Uses LambdaDB list API with pagination; supports both response shapes:
   * { collection, doc } (OpenAPI example) or flat doc object.
   */
  private async getAllDocuments(): Promise<Document[]> {
    try {
      const result = await this.collection.docs.listAll({ size: 100 });
      const documents: Document[] = [];
      for (const item of result.docs ?? []) {
        const raw = item && typeof item === "object" && "doc" in item ? (item as { doc: Record<string, unknown> }).doc : item;
        const doc = lambdaDBToDocument(raw as Record<string, unknown>, this.textField);
        if (raw && typeof raw === "object" && "id" in raw && raw.id !== undefined) {
          doc.metadata.id = raw.id as string;
        }
        documents.push(doc);
      }
      return documents;
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Ensure the collection exists, create if it doesn't
   */
  private async ensureCollectionExists(): Promise<void> {
    try {
      const response = await this.client.listCollections();
      const collectionExists = response.collections?.some(
        (c: { collectionName: string }) => c.collectionName === this.config.collectionName
      );

      if (!collectionExists) {
        await this.createCollection();
      }
    } catch (error) {
      try {
        await this.createCollection();
      } catch (createError) {
        // Collection might already exist (race condition)
      }
    }
  }

  /**
   * Convert LangChain filter to LambdaDB format
   * This is a placeholder - actual implementation depends on LambdaDB's filter syntax
   */
  private convertFilterToLambdaDB(_filter: DocumentFilter): any {
    // This would need to be implemented based on LambdaDB's filter syntax
    // For now, return undefined to handle filtering client-side
    return undefined;
  }
}
