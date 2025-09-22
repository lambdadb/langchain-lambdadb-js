/**
 * Minimal Integration Test for LambdaDB Vector Store
 * 
 * This test focuses on core functionality without complex retry logic.
 */

import { describe, it, expect } from 'vitest';
import { LambdaDBVectorStore } from '../../src/index.js';
import { EmbeddingsInterface } from '@langchain/core/embeddings';

// Minimal embeddings for testing
class MinimalEmbeddings implements EmbeddingsInterface {
  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map(() => [0.8, 0.6, 0.0]);
  }

  async embedQuery(_text: string): Promise<number[]> {
    return [0.8, 0.6, 0.0];
  }
}

describe('LambdaDB Minimal Integration Test', () => {
  it('should perform basic vector store operations', async () => {
    if (!process.env.LAMBDADB_API_KEY) {
      throw new Error('LAMBDADB_API_KEY environment variable is required');
    }

    const embeddings = new MinimalEmbeddings();
    const collectionName = `minimal_test_${Date.now()}`;
    
    // Use minimal configuration to avoid hanging
    const config = {
      projectApiKey: process.env.LAMBDADB_API_KEY!,
      ...(process.env.LAMBDADB_SERVER_URL && { serverURL: process.env.LAMBDADB_SERVER_URL }),
      collectionName,
      vectorDimensions: 3,
      similarityMetric: 'cosine' as const,
      validateCollection: false,
      retryOptions: {
        maxAttempts: 1, // No retries to prevent hanging
        initialDelay: 0,
      },
    };

    const vectorStore = new LambdaDBVectorStore(embeddings, config);
    
    try {
      // Test basic operations
      await vectorStore.createCollection();
      
      const info = await vectorStore.getCollectionInfo();
      expect(info.name).toBe(collectionName);
      expect(info.status).toBe('ACTIVE'); // Should be ACTIVE since createCollection() waits
      
      // Test vector store type
      expect(vectorStore._vectorstoreType()).toBe('lambdadb');
      
      console.log(`✅ Minimal integration test completed for collection: ${collectionName}`);
      
    } finally {
      // Cleanup
      try {
        await vectorStore.deleteCollection();
        console.log(`🧹 Cleaned up collection: ${collectionName}`);
      } catch (error) {
        console.warn(`⚠️ Cleanup warning: ${error.message}`);
      }
    }
  }, 60000); // Increased timeout to accommodate collection creation waiting
});