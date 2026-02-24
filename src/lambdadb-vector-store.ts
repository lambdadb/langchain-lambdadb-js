import { VectorStore } from "@langchain/core/vectorstores";
import { Document } from "@langchain/core/documents";
import { EmbeddingsInterface } from "@langchain/core/embeddings";
import { maximalMarginalRelevance } from "@langchain/core/utils/math";
import { LambdaDBClient } from "@functional-systems/lambdadb";

import {
  LambdaDBConfig,
  CreateCollectionOptions,
  DocumentFilter,
  DeleteOptions,
  MaxMarginalRelevanceSearchOptions,
  CollectionInfo,
  RetryOptions,
  type LambdaDBFilterObject,
} from "./types.js";
import {
  lambdaDBToDocument,
  validateConfig,
  validateVectorDimensions,
  handleLambdaDBError,
  generateDocumentId,
  batchArray,
  withRetry,
  toLambdaDBFilter,
  DEFAULT_RETRY_OPTIONS,
} from "./utils.js";

/**
 * LambdaDB vector store implementation for LangChain
 */
export class LambdaDBVectorStore extends VectorStore {
  declare FilterType: DocumentFilter | LambdaDBFilterObject | string;

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
    
    // Initialize LambdaDB client (0.3.x SDK). Prefer baseUrl + projectName; serverURL supported for backward compatibility.
    this.client = new LambdaDBClient({
      projectApiKey: config.projectApiKey,
      ...(config.baseUrl && { baseUrl: config.baseUrl }),
      ...(config.projectName && { projectName: config.projectName }),
      ...(config.serverURL && { serverURL: config.serverURL }),
      timeoutMs: 30000,
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
   * Add documents to the vector store. Returns document IDs when provided by the store.
   */
  async addDocuments(documents: Document[]): Promise<string[] | void> {
    try {
      if (documents.length === 0) {
        return [];
      }
      const texts = documents.map(({ pageContent }) => pageContent);
      const embeddings = await this.embeddings.embedDocuments(texts);
      return this.addVectors(embeddings, documents);
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Add vectors with associated documents to the store. Returns the assigned document IDs.
   */
  async addVectors(vectors: number[][], documents: Document[]): Promise<string[] | void> {
    try {
      if (vectors.length !== documents.length) {
        throw new Error("Vectors and documents length mismatch");
      }
      if (vectors.length > 0) {
        validateVectorDimensions(vectors[0], this.config.vectorDimensions);
      }
      await this.ensureCollectionExists();

      const lambdaDBDocs = vectors.map((vector, idx) => {
        const doc = documents[idx];
        const id = generateDocumentId();
        return {
          id,
          [this.textField]: doc.pageContent,
          [this.vectorField]: vector,
          ...doc.metadata,
        } as Record<string, unknown>;
      });

      const batchSize = 100;
      const batches = batchArray(lambdaDBDocs, batchSize);
      for (const batch of batches) {
        await withRetry(async () => {
          await this.collection.docs.upsert({ docs: batch });
        }, this.retryOptions);
      }
      return lambdaDBDocs.map((d) => d.id as string);
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Perform similarity search with scores.
   * Filter: object or string → LambdaDB knn.filter (server-side). Function → applied client-side after fetch.
   */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: DocumentFilter | LambdaDBFilterObject | string
  ): Promise<[Document, number][]> {
    try {
      validateVectorDimensions(query, this.config.vectorDimensions);

      const apiFilter = toLambdaDBFilter(filter);
      const knn: Record<string, unknown> = {
        field: this.vectorField,
        queryVector: query,
        k,
      };
      if (apiFilter) {
        knn.filter = apiFilter;
      }

      const response = await withRetry(async () => {
        return await this.collection.query({
          size: k,
          query: { knn },
          consistentRead: this.config.defaultConsistentRead,
        });
      }, this.retryOptions);

      const formattedResults: [Document, number][] = response.docs.map((result) => {
        const doc = lambdaDBToDocument(result.doc, this.textField);
        const score = result.score || 0;
        return [doc, score];
      });

      if (typeof filter === "function") {
        return formattedResults.filter(([doc]) => (filter as DocumentFilter)(doc));
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
    filter?: DocumentFilter | LambdaDBFilterObject | string
  ): Promise<Document[]> {
    const embeddings = await this.embeddings.embedQuery(query);
    const results = await this.similaritySearchVectorWithScore(embeddings, k, filter);
    return results.map(([doc]) => doc);
  }

  /**
   * Create a new collection with vector index.
   * Waits for CREATING → ACTIVE before resolving (LambdaDB creates asynchronously).
   */
  async createCollection(options?: Partial<CreateCollectionOptions>): Promise<void> {
    try {
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
          ...(this.config.partitionConfig || options?.partitionConfig
            ? { partitionConfig: options?.partitionConfig ?? this.config.partitionConfig }
            : {}),
        });
      }, this.retryOptions);

      await this.waitForCollectionActive();
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Wait for collection to become ACTIVE (CREATING → ACTIVE).
   * LambdaDB creates asynchronously; createCollection() uses this so callers see ACTIVE.
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
   * Delete the collection. LambdaDB deletes asynchronously (DELETING → removed);
   * once DELETING, eventual removal is guaranteed. Does not wait for removal.
   * Resolves without throwing if already gone (404) or already DELETING (400 "in DELETING state").
   */
  async deleteCollection(): Promise<void> {
    try {
      await this.collection.delete();
    } catch (error: unknown) {
      const err = error as {
        status?: number;
        statusCode?: number;
        body?: { message?: string };
        message?: string;
      };
      const status = err.status ?? err.statusCode;
      const message = err.body?.message ?? err.message ?? '';
      if (status === 404) return; // already deleted
      if (status === 400 && String(message).includes('DELETING state')) return; // delete already in progress
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
          const apiFilter = toLambdaDBFilter(options.filter);
          if (apiFilter) {
            await this.collection.docs.delete({ filter: apiFilter });
          }
        }
      } else {
        throw new Error("Must provide either ids, filter, or deleteAll option");
      }
    } catch (error) {
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Maximum marginal relevance search: balances relevance to the query with diversity among results.
   * Fetches candidates with includeVectors: true and computes MMR using vector similarity.
   */
  async maxMarginalRelevanceSearch(
    query: string,
    options: MaxMarginalRelevanceSearchOptions,
    _callbacks?: any
  ): Promise<Document[]> {
    const { k = 4, fetchK = 20, lambda = 0.5, filter } = options;

    try {
      const queryVector = await this.embeddings.embedQuery(query);
      validateVectorDimensions(queryVector, this.config.vectorDimensions);

      const apiFilter = toLambdaDBFilter(
        filter as DocumentFilter | LambdaDBFilterObject | string | undefined
      );
      const knn: Record<string, unknown> = {
        field: this.vectorField,
        queryVector,
        k: fetchK,
      };
      if (apiFilter) {
        knn.filter = apiFilter;
      }

      const response = await withRetry(async () => {
        return await this.collection.query({
          size: fetchK,
          query: { knn },
          consistentRead: this.config.defaultConsistentRead,
          includeVectors: true,
        });
      }, this.retryOptions);

      const candidates: { doc: Document; score: number; vector: number[] }[] = [];
      for (const result of response.docs ?? []) {
        const rawDoc = result.doc ?? result;
        const vector = rawDoc[this.vectorField];
        if (!Array.isArray(vector) || vector.length === 0) continue;
        const doc = lambdaDBToDocument(rawDoc as Record<string, unknown>, this.textField);
        const score = typeof result.score === "number" ? result.score : 0;
        candidates.push({ doc, score, vector });
      }

      if (candidates.length === 0) return [];

      const applyClientFilter = typeof filter === "function";
      const list = applyClientFilter
        ? candidates.filter((c) => (filter as DocumentFilter)(c.doc))
        : candidates;
      if (list.length === 0) return [];

      const embeddingList = list.map((c) => c.vector);
      const selectedIndexes = maximalMarginalRelevance(queryVector, embeddingList, lambda, k);
      return selectedIndexes.map((i) => list[i].doc);
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
