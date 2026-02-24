/**
 * Diagnostic test to check LambdaDB API connectivity
 *
 * Uses the LambdaDB 0.3.x client (LambdaDBClient) to verify connection.
 */

import { describe, it, expect } from 'vitest';
import { LambdaDBClient } from '@functional-systems/lambdadb';

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

      // Clean up
      try {
        await collection.delete();
        console.log('🧹 Collection cleanup succeeded!');
      } catch (cleanupError: unknown) {
        console.warn('⚠️ Collection cleanup failed:', (cleanupError as Error).message);
      }
      
    } catch (error) {
      console.error('❌ Collection creation failed:');
      console.error('Error name:', error.name);
      console.error('Error message:', error.message);
      console.error('Error status:', error.status || error.statusCode);
      console.error('Error body:', error.body);
      throw error;
    }
  }, 30000);
});