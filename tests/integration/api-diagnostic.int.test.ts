/**
 * Diagnostic test to check LambdaDB API connectivity
 *
 * Uses the LambdaDB 0.3.x client (LambdaDBClient) to verify connection.
 * LambdaDB creates/deletes asynchronously: CREATING → ACTIVE (we wait), DELETING → removed (guaranteed; we do not wait).
 */

import { describe, it, expect } from 'vitest';
import { LambdaDBClient } from '@functional-systems/lambdadb';

/** Poll until collection status is ACTIVE (LambdaDB creates asynchronously). */
async function waitForCollectionActive(
  client: LambdaDBClient,
  collectionName: string,
  maxWaitMs = 30000,
  pollIntervalMs = 1000
): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    const list = await client.listCollections();
    const col = list.collections?.find((c: { collectionName?: string }) => c.collectionName === collectionName);
    const status = (col as { collectionStatus?: string } | undefined)?.collectionStatus;
    if (status === 'ACTIVE') return;
    await new Promise((r) => setTimeout(r, pollIntervalMs));
  }
  throw new Error(`Collection did not become ACTIVE within ${maxWaitMs}ms`);
}

describe('LambdaDB API Diagnostic', () => {
  it('should connect to LambdaDB and list collections', async () => {
    if (!process.env.LAMBDADB_API_KEY) {
      throw new Error('LAMBDADB_API_KEY environment variable is required');
    }

    const client = new LambdaDBClient({
      projectApiKey: process.env.LAMBDADB_API_KEY!,
      ...(process.env.LAMBDADB_SERVER_URL && { serverURL: process.env.LAMBDADB_SERVER_URL }),
      timeoutMs: 10000
    });
    console.log('🔍 Testing LambdaDB API connectivity...');
    console.log('📡 API Key:', process.env.LAMBDADB_API_KEY?.slice(0, 10) + '...');
    console.log('🌐 Server URL:', process.env.LAMBDADB_SERVER_URL || 'default');

    try {
      const response = await client.listCollections();
      console.log('✅ Successfully connected to LambdaDB!');
      console.log('📋 Collections response:', JSON.stringify(response, null, 2));
      
      expect(response).toBeDefined();
    } catch (error) {
      console.error('❌ LambdaDB API connection failed:');
      console.error('Error name:', error.name);
      console.error('Error message:', error.message);
      console.error('Error status:', error.status || error.statusCode);
      console.error('Full error:', error);
      throw error;
    }
  }, 30000);

  it('should handle collection creation attempt', async () => {
    if (!process.env.LAMBDADB_API_KEY) {
      throw new Error('LAMBDADB_API_KEY environment variable is required');
    }

    const client = new LambdaDBClient({
      projectApiKey: process.env.LAMBDADB_API_KEY!,
      ...(process.env.LAMBDADB_SERVER_URL && { serverURL: process.env.LAMBDADB_SERVER_URL }),
      timeoutMs: 10000
    });

    const testCollectionName = `diagnostic_test_${Date.now()}`;
    const collection = client.collection(testCollectionName);

    try {
      console.log(`🔨 Attempting to create collection: ${testCollectionName}`);

      await client.createCollection({
        collectionName: testCollectionName,
        indexConfigs: {
          embedding: {
            type: "vector",
            dimensions: 3,
            similarity: "cosine",
          },
        },
      });

      console.log('✅ Collection creation succeeded!');

      // Wait for CREATING → ACTIVE before delete (LambdaDB does not allow delete while CREATING)
      await waitForCollectionActive(client, testCollectionName);
      console.log('✅ Collection is ACTIVE, proceeding to delete.');

      await collection.delete();
      console.log('🧹 Collection delete requested (DELETING → removal is guaranteed by LambdaDB).');
      
    } catch (error) {
      console.error('❌ Collection creation failed:');
      console.error('Error name:', error.name);
      console.error('Error message:', error.message);
      console.error('Error status:', error.status || error.statusCode);
      console.error('Error body:', error.body);
      throw error;
    }
  }, 60000); // CREATING→ACTIVE + delete
});