import { VectorStore } from "@langchain/core/vectorstores";
import { Document } from "@langchain/core/documents";
import { EmbeddingsInterface } from "@langchain/core/embeddings";
import { maximalMarginalRelevance } from "@langchain/core/utils/math";

import {
  LambdaDBVectorStoreConfig,
  DocumentFilter,
  DeleteOptions,
  MaxMarginalRelevanceSearchOptions,
  CollectionInfo,
  type LambdaDBFilterObject,
  type VectorSearchOptions,
} from "./types.js";
import {
  lambdaDBToDocument,
  validateConfig,
  validateVectorDimensions,
  handleLambdaDBError,
  generateDocumentId,
  UPSERT_PAYLOAD_SIZE_THRESHOLD_BYTES,
  toLambdaDBFilter,
} from "./utils.js";

/**
 * LambdaDB vector store implementation for LangChain.
 * Constructor and static methods match base VectorStore signatures; pass collection via config.collection.
 */
export class LambdaDBVectorStore extends VectorStore {
  declare FilterType: DocumentFilter | LambdaDBFilterObject | string;

  private collection: LambdaDBVectorStoreConfig["collection"];
  private config: LambdaDBVectorStoreConfig;
  private textField: string;
  private vectorField: string;
  private _vectorDimensions: number | null = null;

  constructor(embeddings: EmbeddingsInterface, config: LambdaDBVectorStoreConfig) {
    super(embeddings, config);

    validateConfig(config);

    this.collection = config.collection;
    // Set configuration with defaults
    this.config = {
      textField: "page_content",
      vectorField: "vector",
      defaultConsistentRead: false,
      ...config,
    };

    this.textField = this.config.textField!;
    this.vectorField = this.config.vectorField!;
  }

  /** Throw a clear error if the failure is due to collection not existing (404). */
  private throwIfCollectionNotFound(error: unknown): void {
    const err = error as { status?: number; statusCode?: number };
    if (err.status === 404 || err.statusCode === 404) {
      const name = (this.collection as { collectionName?: string }).collectionName;
      const namePart = name ? ` '${name}'` : "";
      throw new Error(
        `Collection${namePart} does not exist. Create it first using the LambdaDB client (e.g. client.createCollection({ collectionName, indexConfigs: { ... } })), then try again.`
      );
    }
  }

  /**
   * Get vector dimension from embeddings (cached after first call).
   */
  private async getVectorDimensions(): Promise<number> {
    if (this._vectorDimensions != null) {
      return this._vectorDimensions;
    }
    const vec = await this.embeddings.embedQuery("x");
    this._vectorDimensions = vec.length;
    return this._vectorDimensions;
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
        validateVectorDimensions(vectors[0], await this.getVectorDimensions());
      }

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

      const payloadSize = new TextEncoder().encode(JSON.stringify(lambdaDBDocs)).length;
      if (payloadSize <= UPSERT_PAYLOAD_SIZE_THRESHOLD_BYTES) {
        await this.collection.docs.upsert({ docs: lambdaDBDocs });
      } else {
        await this.collection.docs.bulkUpsertDocs({ docs: lambdaDBDocs });
      }
      return lambdaDBDocs.map((d) => d.id as string);
    } catch (error: unknown) {
      this.throwIfCollectionNotFound(error);
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Perform similarity search with scores.
   * Filter: object or string → LambdaDB knn.filter (server-side). Function → applied client-side after fetch.
   * options.consistentRead overrides defaultConsistentRead for this call.
   */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: DocumentFilter | LambdaDBFilterObject | string,
    options?: VectorSearchOptions
  ): Promise<[Document, number][]> {
    try {
      validateVectorDimensions(query, await this.getVectorDimensions());

      const apiFilter = toLambdaDBFilter(filter);
      const knn: Record<string, unknown> = {
        field: this.vectorField,
        queryVector: query,
        k,
      };
      if (apiFilter) {
        knn.filter = apiFilter;
      }

      const consistentRead = options?.consistentRead ?? this.config.defaultConsistentRead;
      const response = await this.collection.query({
        size: k,
        query: { knn },
        consistentRead,
      });

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
      this.throwIfCollectionNotFound(error);
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Perform similarity search without scores.
   * options.consistentRead overrides defaultConsistentRead for this call.
   */
  async similaritySearch(
    query: string,
    k = 4,
    filter?: DocumentFilter | LambdaDBFilterObject | string,
    _callbacks?: unknown,
    options?: VectorSearchOptions
  ): Promise<Document[]> {
    const embeddings = await this.embeddings.embedQuery(query);
    const results = await this.similaritySearchVectorWithScore(embeddings, k, filter, options);
    return results.map(([doc]) => doc);
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
      this.throwIfCollectionNotFound(error);
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
    const consistentRead = options.consistentRead ?? this.config.defaultConsistentRead;

    try {
      const queryVector = await this.embeddings.embedQuery(query);
      validateVectorDimensions(queryVector, await this.getVectorDimensions());

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

      const response = await this.collection.query({
        size: fetchK,
        query: { knn },
        consistentRead,
        includeVectors: true,
      });

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
      this.throwIfCollectionNotFound(error);
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
        name: col.collectionName ?? (this.collection as { collectionName?: string }).collectionName ?? "",
        status: col.collectionStatus ?? "unknown",
        documentCount: col.numDocs,
        indexConfigs: col.indexConfigs,
        createdAt: undefined,
        updatedAt: undefined,
      };
    } catch (error) {
      this.throwIfCollectionNotFound(error);
      throw handleLambdaDBError(error);
    }
  }

  /**
   * Static method to create LambdaDBVectorStore from texts.
   * Pass client via config.client to reuse the same client across collections.
   */
  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: EmbeddingsInterface,
    config: LambdaDBVectorStoreConfig
  ): Promise<LambdaDBVectorStore> {
    const docs = texts.map((text, idx) => {
      const metadata = Array.isArray(metadatas) ? metadatas[idx] || {} : metadatas || {};
      return new Document({ pageContent: text, metadata });
    });

    return LambdaDBVectorStore.fromDocuments(docs, embeddings, config);
  }

  /**
   * Static method to create LambdaDBVectorStore from documents.
   * Pass client via config.client to reuse the same client across collections.
   */
  static async fromDocuments(
    docs: Document[],
    embeddings: EmbeddingsInterface,
    config: LambdaDBVectorStoreConfig
  ): Promise<LambdaDBVectorStore> {
    const instance = new LambdaDBVectorStore(embeddings, config);
    await instance.addDocuments(docs);
    return instance;
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
      this.throwIfCollectionNotFound(error);
      throw handleLambdaDBError(error);
    }
  }
}
