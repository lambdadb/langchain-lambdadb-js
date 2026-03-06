import { Document } from "@langchain/core/documents";
import type { LambdaDBFilterObject, LambdaDBVectorStoreConfig } from "./types.js";

/**
 * Convert LambdaDB document back to LangChain Document
 */
export function lambdaDBToDocument(
  lambdaDoc: any, 
  textField: string = "page_content"
): Document {
  // Extract text content from the specified field
  const pageContent = lambdaDoc[textField] || lambdaDoc.page_content || lambdaDoc.pageContent || lambdaDoc.content || "";
  
  // Extract metadata (exclude vector field and text field from metadata)
  const metadata = { ...lambdaDoc };
  delete metadata[textField];
  delete metadata.embedding;  // Remove default vector field
  delete metadata.vector;     // Remove alternative vector field names
  delete metadata.id;         // Remove document ID from metadata
  
  return new Document({
    pageContent,
    metadata,
  });
}

/**
 * Generate a unique ID for a document if not provided
 */
export function generateDocumentId(): string {
  return `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Validate vector dimensions match expected dimensions
 */
export function validateVectorDimensions(
  vector: number[],
  expectedDimensions: number
): void {
  if (vector.length !== expectedDimensions) {
    throw new Error(
      `Vector dimension mismatch: expected ${expectedDimensions}, got ${vector.length}`
    );
  }
}

/**
 * Validate vector store configuration parameters.
 * Dimension is derived from embeddings (cached on first use).
 */
export function validateConfig(config: LambdaDBVectorStoreConfig): void {
  if (config.collection == null) {
    throw new Error("collection is required");
  }
}

/**
 * Enhanced LambdaDB error types
 */
export class LambdaDBConnectionError extends Error {
  constructor(message: string, public originalError?: any) {
    super(message);
    this.name = 'LambdaDBConnectionError';
  }
}

export class LambdaDBAuthenticationError extends Error {
  constructor(message: string, public originalError?: any) {
    super(message);
    this.name = 'LambdaDBAuthenticationError';
  }
}

export class LambdaDBResourceNotFoundError extends Error {
  constructor(message: string, public originalError?: any) {
    super(message);
    this.name = 'LambdaDBResourceNotFoundError';
  }
}

export class LambdaDBValidationError extends Error {
  constructor(message: string, public originalError?: any) {
    super(message);
    this.name = 'LambdaDBValidationError';
  }
}

export class LambdaDBRateLimitError extends Error {
  constructor(message: string, public retryAfter?: number, public originalError?: any) {
    super(message);
    this.name = 'LambdaDBRateLimitError';
    this.retryAfter = retryAfter;
  }
}

/**
 * Handle LambdaDB errors and convert to specific error types
 */
export function handleLambdaDBError(error: any): Error {
  // Handle HTTP status codes
  if (error.status || error.statusCode) {
    const status = error.status || error.statusCode;
    const message = error.message || error.body?.message || 'Unknown error';
    
    switch (status) {
      case 401:
      case 403:
        return new LambdaDBAuthenticationError(
          `Authentication failed: ${message}`, 
          error
        );
      case 404:
        return new LambdaDBResourceNotFoundError(
          `Resource not found: ${message}`, 
          error
        );
      case 400:
        return new LambdaDBValidationError(
          `Validation error: ${message}`, 
          error
        );
      case 429:
        const retryAfter = error.headers?.['retry-after'] 
          ? parseInt(error.headers['retry-after']) 
          : undefined;
        return new LambdaDBRateLimitError(
          `Rate limit exceeded: ${message}`, 
          retryAfter,
          error
        );
      case 500:
      case 502:
      case 503:
      case 504:
        return new LambdaDBConnectionError(
          `Server error: ${message}`, 
          error
        );
    }
  }

  // Handle specific error names
  if (error.name === 'LambdaDBError' || error.name === 'UnauthenticatedError') {
    return new LambdaDBAuthenticationError(`LambdaDB Error: ${error.message}`, error);
  }
  if (error.name === 'ResourceNotFoundError') {
    return new LambdaDBResourceNotFoundError(`Resource not found: ${error.message}`, error);
  }
  
  // Handle network errors
  if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
    return new LambdaDBConnectionError(
      `Connection failed: ${error.message}`, 
      error
    );
  }

  return error instanceof Error ? error : new Error(String(error));
}

/**
 * Cosine similarity between two vectors (returns value in [-1, 1]).
 * Used for MMR diversity/relevance balance.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Payload size threshold (1MB). Below this the vector store uses a single upsert; above it uses bulkUpsertDocs.
 * Aligns with Python implementation: UPSERT_PAYLOAD_SIZE_THRESHOLD_BYTES.
 */
export const UPSERT_PAYLOAD_SIZE_THRESHOLD_BYTES = 1024 * 1024;

/**
 * Convert a filter (string, LambdaDB object, or function) to LambdaDB API filter for query/delete.
 * Used for server-side filtering in search (knn.filter) and delete (docs.delete filter).
 *
 * @param filter - Query string (e.g. "field:value"), LambdaDB filter object (e.g. { queryString: { query: "..." } }), or function (client-side only).
 * @returns LambdaDB filter object for API, or undefined when filter is a function or null/undefined.
 * @see https://docs.lambdadb.ai/guides/search/query-string
 * @see https://docs.lambdadb.ai/guides/documents/delete-data
 */
export function toLambdaDBFilter(
  filter: ((doc: Document) => boolean) | LambdaDBFilterObject | string | undefined | null
): LambdaDBFilterObject | undefined {
  if (filter == null) return undefined;
  if (typeof filter === "function") return undefined;
  if (typeof filter === "string") return { queryString: { query: filter } };
  if (typeof filter === "object" && filter !== null) return filter as LambdaDBFilterObject;
  return undefined;
}