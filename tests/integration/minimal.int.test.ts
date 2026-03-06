/**
 * Minimal Integration Test for LambdaDB Vector Store
 *
 * Uses defaultConsistentRead: true so query/fetch see writes immediately (LambdaDB is eventually consistent by default).
 * Collections are created/deleted via the LambdaDB client.
 */

import { describe, it, expect } from 'vitest';
import { LambdaDBClient } from '@functional-systems/lambdadb';
import { LambdaDBVectorStore } from '../../src/index.js';
import { EmbeddingsInterface } from '@langchain/core/embeddings';

const VECTOR_DIMENSIONS = 3;

// Minimal embeddings for testing
class MinimalEmbeddings implements EmbeddingsInterface {
  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map(() => [0.8, 0.6, 0.0]);
  }

  async embedQuery(_text: string): Promise<number[]> {
    return [0.8, 0.6, 0.0];
  }
}

async function createTestCollection(client: LambdaDBClient, collectionName: string): Promise<void> {
  await client.createCollection({
    collectionName,
    indexConfigs: {
      vector: { type: 'vector', dimensions: VECTOR_DIMENSIONS, similarity: 'cosine' },
      page_content: { type: 'text', analyzers: ['english'] },
    },
  });
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const list = await client.listCollections();
    const col = (list as { collections?: Array<{ collectionName?: string; collectionStatus?: string }> }).collections?.find(
      (c) => c.collectionName === collectionName
    );
    if (col?.collectionStatus === 'ACTIVE') return;
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`Collection ${collectionName} did not become ACTIVE in time`);
}

describe('LambdaDB Minimal Integration Test', () => {
  it('should perform basic vector store operations', async () => {
    if (!process.env.LAMBDADB_PROJECT_API_KEY) {
      throw new Error('LAMBDADB_PROJECT_API_KEY environment variable is required');
    }

    const client = new LambdaDBClient({
      projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!,
      baseUrl: process.env.LAMBDADB_BASE_URL,
      projectName: process.env.LAMBDADB_PROJECT_NAME,
      timeoutMs: 10000,
    });
    const embeddings = new MinimalEmbeddings();
    const collectionName = `minimal_test_${Date.now()}`;

    await createTestCollection(client, collectionName);

    const vectorStore = new LambdaDBVectorStore(embeddings, {
      collection: client.collection(collectionName),
      defaultConsistentRead: true,
    });

    try {
      const info = await vectorStore.getCollectionInfo();
      expect(info.name).toBe(collectionName);
      expect(info.status).toBe('ACTIVE');

      expect(vectorStore._vectorstoreType()).toBe('lambdadb');

      console.log(`✅ Minimal integration test completed for collection: ${collectionName}`);
    } finally {
      try {
        await client.collection(collectionName).delete();
        console.log(`🧹 Cleaned up collection: ${collectionName}`);
      } catch (error) {
        console.warn(`⚠️ Cleanup warning: ${(error as Error).message}`);
      }
    }
  }, 60000);
});