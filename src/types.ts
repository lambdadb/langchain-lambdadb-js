import { Document } from "@langchain/core/documents";

/**
 * Retry configuration options
 */
export interface RetryOptions {
  maxAttempts: number;
  initialDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  retryableErrors: string[];
}

/**
 * Supported similarity metrics for LambdaDB vector search
 */
export type SimilarityMetric = 'cosine' | 'euclidean' | 'dot_product' | 'max_inner_product';

/**
 * Configuration options for LambdaDB vector store
 */
export interface LambdaDBConfig {
  /** LambdaDB project API key */
  projectApiKey: string;
  /** Custom server URL (optional) */
  serverURL?: string;
  /** Name of the collection to use */
  collectionName: string;
  /** Vector dimensions for the embeddings */
  vectorDimensions: number;
  /** Similarity metric to use for vector comparisons */
  similarityMetric?: SimilarityMetric;
  /** Custom index configuration */
  indexConfig?: Record<string, any>;
  /** Name of the text field in documents (default: "content") */
  textField?: string;
  /** Name of the vector field in documents (default: "embedding") */
  vectorField?: string;
  /** Whether to validate that the collection exists on initialization */
  validateCollection?: boolean;
  /** Default setting for consistent reads (default: false) */
  defaultConsistentRead?: boolean;
  /** Retry configuration for failed operations */
  retryOptions?: Partial<RetryOptions>;
}

/**
 * Options for creating a new collection
 */
export interface CreateCollectionOptions {
  /** Collection name */
  name: string;
  /** Vector field configuration */
  vectorConfig: {
    dimensions: number;
    similarity: SimilarityMetric;
  };
  /** Additional index configuration */
  indexConfig?: Record<string, any>;
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
}

/**
 * Delete operation options
 */
export interface DeleteOptions {
  /** Document IDs to delete */
  ids?: string[];
  /** Filter function to select documents for deletion */
  filter?: DocumentFilter;
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
