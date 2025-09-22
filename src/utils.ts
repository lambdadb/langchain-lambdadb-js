import { Document } from "@langchain/core/documents";
import { LambdaDBDocument, SimilaritySearchResult } from "./types.js";

/**
 * Convert LangChain Document to LambdaDB document format
 */
export function documentToLambdaDB(
  doc: Document,
  embedding: number[],
  id?: string
): LambdaDBDocument {
  return {
    id,
    content: doc.pageContent,
    embedding,
    metadata: doc.metadata || {},
  };
}

/**
 * Convert LambdaDB document back to LangChain Document
 */
export function lambdaDBToDocument(
  lambdaDoc: any, 
  textField: string = "content"
): Document {
  // Extract text content from the specified field
  const pageContent = lambdaDoc[textField] || lambdaDoc.content || lambdaDoc.pageContent || "";
  
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
 * Convert search results with scores to the expected format
 */
export function formatSearchResults(
  results: any[],
  includeScores: boolean = true
): SimilaritySearchResult[] {
  return results.map((result) => ({
    document: lambdaDBToDocument(result),
    score: includeScores && result._score ? result._score : 0,
  }));
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
 * Validate configuration parameters
 */
export function validateConfig(config: {
  projectApiKey: string;
  collectionName: string;
  vectorDimensions: number;
}): void {
  if (!config.projectApiKey) {
    throw new Error("projectApiKey is required");
  }
  if (!config.collectionName) {
    throw new Error("collectionName is required");
  }
  if (!config.vectorDimensions || config.vectorDimensions <= 0) {
    throw new Error("vectorDimensions must be a positive number");
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
 * Default retry configuration
 */
export const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxAttempts: 3,
  initialDelay: 1000,
  maxDelay: 10000,
  backoffMultiplier: 2,
  retryableErrors: [
    'LambdaDBConnectionError',
    'LambdaDBRateLimitError',
    'ECONNREFUSED',
    'ENOTFOUND',
    'TIMEOUT'
  ],
};

/**
 * Execute function with retry logic
 */
export async function withRetry<T>(
  fn: () => Promise<T>, 
  options: Partial<RetryOptions> = {}
): Promise<T> {
  const config = { ...DEFAULT_RETRY_OPTIONS, ...options };
  let lastError: Error;
  
  for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      
      // Don't retry if this is the last attempt
      if (attempt === config.maxAttempts) {
        break;
      }
      
      // Check if error is retryable
      const isRetryable = config.retryableErrors.some(errorType => 
        lastError.name === errorType || 
        lastError.message.includes(errorType) ||
        (lastError as any).code === errorType
      );
      
      if (!isRetryable) {
        break;
      }
      
      // Calculate delay with exponential backoff
      let delay = config.initialDelay * Math.pow(config.backoffMultiplier, attempt - 1);
      
      // Handle rate limit specific delay
      if (lastError instanceof LambdaDBRateLimitError && lastError.retryAfter) {
        delay = lastError.retryAfter * 1000; // Convert seconds to milliseconds
      }
      
      delay = Math.min(delay, config.maxDelay);
      
      console.warn(`LambdaDB operation failed (attempt ${attempt}/${config.maxAttempts}), retrying in ${delay}ms:`, lastError.message);
      
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  throw lastError!;
}

/**
 * Batch array into smaller chunks for processing
 */
export function batchArray<T>(array: T[], batchSize: number): T[][] {
  const batches: T[][] = [];
  for (let i = 0; i < array.length; i += batchSize) {
    batches.push(array.slice(i, i + batchSize));
  }
  return batches;
}