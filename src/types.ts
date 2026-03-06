import { Document } from "@langchain/core/documents";
import type { LambdaDBClient } from "@functional-systems/lambdadb";

/** Collection handle type (returned by client.collection(name)); has .collectionName. */
export type LambdaDBCollectionHandle = ReturnType<LambdaDBClient["collection"]>;

/**
 * Supported similarity metrics for LambdaDB vector search
 */
export type SimilarityMetric = 'cosine' | 'euclidean' | 'dot_product' | 'max_inner_product';

/** Partition config for collection (LambdaDB API). Optional; for future use. */
export interface PartitionConfigOption {
  fieldName?: string;
  dataType?: "keyword";
  numPartitions?: number;
}

/**
 * Configuration options for LambdaDB vector store.
 * Pass a collection handle (client.collection(name)); create/delete collections via the LambdaDB client.
 */
export interface LambdaDBVectorStoreConfig {
  /** LambdaDB collection handle (e.g. client.collection('my-docs')). */
  collection: LambdaDBCollectionHandle;
  /** Name of the text field in documents (default: "page_content") */
  textField?: string;
  /** Name of the vector field in documents (default: "vector") */
  vectorField?: string;
  /** Use consistent reads for query/fetch (default: false). Set true when you need to see writes immediately; otherwise LambdaDB uses eventual consistency. */
  defaultConsistentRead?: boolean;
}

/**
 * Options for similarity search
 */
export interface SimilaritySearchOptions {
  /** Number of results to return */
  k: number;
  /** Filter function for documents */
  filter?: (doc: Document) => boolean;
  /** Additional query parameters */
  queryParams?: Record<string, any>;
}

/**
 * Result from similarity search with score
 */
export interface SimilaritySearchResult {
  /** The document */
  document: Document;
  /** Similarity score */
  score: number;
}

/**
 * LambdaDB document structure for vector storage
 */
export interface LambdaDBDocument {
  /** Unique document ID */
  id?: string;
  /** Document content */
  content: string;
  /** Vector embedding */
  embedding: number[];
  /** Document metadata */
  metadata: Record<string, any>;
}

/**
 * Query options for LambdaDB vector search
 */
export interface QueryOptions {
  /** Query vector */
  vector: number[];
  /** Number of results to return */
  k: number;
  /** Metadata filters */
  filter?: Record<string, any>;
  /** Include similarity scores in results */
  includeScores?: boolean;
}

/**
 * Filter type for document filtering
 */
export type DocumentFilter = (doc: Document) => boolean;

/**
 * Optional per-call override for search methods (e.g. consistentRead).
 */
export interface VectorSearchOptions {
  /** Override defaultConsistentRead for this call. When true, query uses consistent read. */
  consistentRead?: boolean;
}

/**
 * Options for maximum marginal relevance search
 */
export interface MaxMarginalRelevanceSearchOptions {
  /** Number of results to return (default: 4) */
  k?: number;
  /** Number of candidates to fetch initially (default: 20) */
  fetchK?: number;
  /** Diversity factor (0 = max diversity, 1 = max relevance) (default: 0.5) */
  lambda?: number;
  /** Filter function for documents */
  filter?: DocumentFilter | string | object;
  /** Override defaultConsistentRead for this call. When true, query uses consistent read. */
  consistentRead?: boolean;
}

/**
 * LambdaDB filter object for server-side delete/query.
 * Use LambdaDB query syntax, e.g. { queryString: { query: "field:value" } }.
 * @see https://docs.lambdadb.ai/guides/documents/delete-data
 * @see https://docs.lambdadb.ai/guides/search/query-string
 */
export type LambdaDBFilterObject = Record<string, unknown>;

/**
 * Delete operation options
 */
export interface DeleteOptions {
  /** Document IDs to delete */
  ids?: string[];
  /**
   * Filter for which documents to delete. Prefer LambdaDB filter for efficiency (server-side).
   * - Object: LambdaDB filter, e.g. { queryString: { query: "genre:documentary" } } → passed to API as-is.
   * - String: treated as query string, e.g. "genre:documentary" → { queryString: { query } }.
   * - Function: client-side filter (doc) => boolean; fetches all docs then deletes by ids (less efficient).
   */
  filter?: DocumentFilter | LambdaDBFilterObject | string;
  /** Whether to delete all documents in collection */
  deleteAll?: boolean;
}

/**
 * Enhanced search options for similarity search
 */
export interface EnhancedSimilaritySearchOptions extends SimilaritySearchOptions {
  /** Whether to use consistent read */
  consistentRead?: boolean;
  /** Include document vectors in response */
  includeVectors?: boolean;
  /** Custom fields to include in response */
  fields?: string[];
}

/**
 * Collection information
 */
export interface CollectionInfo {
  /** Collection name */
  name: string;
  /** Collection status */
  status: string;
  /** Number of documents */
  documentCount?: number;
  /** Index configurations */
  indexConfigs?: Record<string, any>;
  /** Creation timestamp */
  createdAt?: string;
  /** Last updated timestamp */
  updatedAt?: string;
}
