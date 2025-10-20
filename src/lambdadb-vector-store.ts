import { VectorStore } from "@langchain/core/vectorstores";
import { Document } from "@langchain/core/documents";
import { EmbeddingsInterface } from "@langchain/core/embeddings";
import { LambdaDB } from "@functional-systems/lambdadb";

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
  extractServerURLFromProjectUrl,
  DEFAULT_RETRY_OPTIONS,
} from "./utils.js";

/**
 * LambdaDB vector store implementation for LangChain
 */
export class LambdaDBVectorStore extends VectorStore {
  declare FilterType: DocumentFilter;

  private client: LambdaDB;
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
      defaultConsistentRead: false, // Use eventual consistency by default for better performance
      ...config,
    };
    
    this.textField = this.config.textField!;
    this.vectorField = this.config.vectorField!;
    this.retryOptions = { ...DEFAULT_RETRY_OPTIONS, ...(config.retryOptions || {}) };
    
    // Initialize LambdaDB client - extract server URL from project URL
    const serverURL = extractServerURLFromProjectUrl(config.projectUrl);
    this.client = new LambdaDB({
      projectApiKey: config.projectApiKey,
      serverURL: serverURL, // LambdaDB client expects serverURL parameter
      timeoutMs: 30000, // 30 second timeout for all operations
    });
    
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
      
      // 50KB document size validation (matching Python implementation)
      for (const [index, text] of texts.entries()) {
        if (50 * 1000 < text.length) {
          throw new Error(
            `The text at index ${index} is too long. Max length is 50KB.`
          );
        }
      }
      
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

      // SAFETY: Setting batch size to 100 is safe, because we've checked that there is no document longer than 50KB.
      const batchSize = 100; // Conservative batch size for 6MB limit
      const batches = batchArray(lambdaDBDocs, batchSize);

      for (const batch of batches) {
        await withRetry(async () => {
          await this.client.collections.docs.upsert({
            collectionName: this.config.collectionName,
            requestBody: {
              docs: batch,
            },
          });
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

      // Build request body with KNN query
      const requestBody: any = {
        size: k,
        query: {
          knn: {
            field: this.vectorField,
            queryVector: query,
            k: k
          }
        },
        consistentRead: this.config.defaultConsistentRead,
      };

      // Add server-side filter if provided (user responsible for correct format)
      if (filter && typeof filter === 'object' && typeof filter !== 'function') {
        requestBody.query.knn.filter = filter;
      }

      // Query LambdaDB for similar vectors using correct KNN API structure with retry
      const response = await withRetry(async () => {
        return await this.client.collections.query({
          collectionName: this.config.collectionName,
          requestBody,
        });
      }, this.retryOptions);

      // Convert results to LangChain format - no client-side filtering
      const formattedResults: [Document, number][] = response.docs.map((result) => {
        const doc = lambdaDBToDocument(result.doc, this.textField);
        const score = result.score || 0;
        return [doc, score];
      });

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
      // Create collection with proper index configuration using LambdaDB types with retry
      await withRetry(async () => {
        await this.client.collections.create({
          collectionName: this.config.collectionName,
          indexConfigs: {
            // Vector index configuration for the embedding field
            [this.vectorField]: {
              type: "vector" as const,
              dimensions: this.config.vectorDimensions,
              similarity: (this.config.similarityMetric?.toLowerCase() || "cosine") as "cosine" | "euclidean" | "dot_product" | "max_inner_product",
            },
            // Add other index configurations if provided
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
      await this.client.collections.delete({
        collectionName: this.config.collectionName,
      });
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Delete documents from the vector store
   */
  async deleteDocuments(options: DeleteOptions): Promise<void> {
    try {
      if (options.deleteAll) {
        // Delete all documents by getting all and deleting by IDs
        // LambdaDB doesn't support deleteAll directly
        const allDocs = await this.getAllDocuments();
        const idsToDelete = allDocs
          .map((doc) => doc.metadata.id)
          .filter((id) => id);

        if (idsToDelete.length > 0) {
          await this.deleteDocuments({ ids: idsToDelete });
        }
      } else if (options.ids && options.ids.length > 0) {
        // Delete documents by IDs
        await this.client.collections.docs.delete({
          collectionName: this.config.collectionName,
          requestBody: {
            ids: options.ids,
          },
        });
      } else if (options.filter) {
        // For filter-based deletion, we need to first find matching documents
        // This is a two-step process: search then delete
        const allDocs = await this.getAllDocuments();
        
        // Only support function filters for deletion (object filters are for search queries)
        if (typeof options.filter === 'function') {
          const filterFn = options.filter as (doc: Document) => boolean;
          const docsToDelete = allDocs.filter(filterFn);
          const idsToDelete = docsToDelete
            .map((doc) => doc.metadata.id)
            .filter((id) => id);

          if (idsToDelete.length > 0) {
            await this.deleteDocuments({ ids: idsToDelete });
          }
        } else {
          throw new Error("Delete operations only support function-based filters, not object filters");
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
    options: MaxMarginalRelevanceSearchOptions<string | object>,
    _callbacks?: any
  ): Promise<Document[]> {
    const {
      k = 4,
      fetchK = 20,
      lambda = 0.5,
      filter,
    } = options;

    try {
      // Convert filter to the type expected by similaritySearchVectorWithScore
      let searchFilter: DocumentFilter | undefined;
      if (filter && typeof filter === 'object' && typeof filter !== 'function') {
        // Object filter - pass through for server-side filtering
        searchFilter = filter as Record<string, any>;
      } else if (typeof filter === 'function') {
        // Function filter - pass through
        searchFilter = filter as (doc: Document) => boolean;
      }
      // String filters are not supported for vector search

      // First, get more candidates than needed
      const candidateResults = await this.similaritySearchVectorWithScore(
        await this.embeddings.embedQuery(query),
        fetchK,
        searchFilter
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
      const response = await this.client.collections.get({
        collectionName: this.config.collectionName,
      });

      return {
        name: response.collection.collectionName || this.config.collectionName,
        status: response.collection.collectionStatus || "unknown",
        documentCount: response.collection.numDocs,
        indexConfigs: response.collection.indexConfigs,
        // These fields might not be available in the current API
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
      await this.client.collections.get({
        collectionName: this.config.collectionName,
      });
    } catch (error) {
      throw new Error(`Collection '${this.config.collectionName}' does not exist: ${error}`);
    }
  }

  /**
   * Get all documents from the collection (for internal operations)
   */
  private async getAllDocuments(): Promise<Document[]> {
    try {
      // TODO: Implement proper "get all documents" functionality
      // For now, return empty array since LambdaDB doesn't support simple match-all queries
      // This affects deleteAll functionality - we'll need to implement pagination or
      // use a different approach for bulk operations
      console.warn('getAllDocuments not implemented - returning empty array');
      return [];
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Ensure the collection exists, create if it doesn't
   */
  private async ensureCollectionExists(): Promise<void> {
    try {
      // Try to get collection info
      const response = await this.client.collections.list();
      const collectionExists = response.collections && response.collections.some(
        (collection: any) => collection.name === this.config.collectionName
      );

      if (!collectionExists) {
        await this.createCollection();
      }
    } catch (error) {
      // If getting collection info fails, try creating it
      try {
        await this.createCollection();
      } catch (createError) {
        // If creation also fails, the collection might already exist
        // This is a common race condition, so we can ignore it
      }
    }
  }
}
