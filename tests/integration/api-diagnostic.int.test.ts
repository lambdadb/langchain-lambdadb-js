/**
 * Diagnostic test to check LambdaDB API connectivity
 * 
 * This test directly uses the LambdaDB client to verify connection
 * without going through our vector store implementation.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { LambdaDB } from '@functional-systems/lambdadb';
import { extractServerURLFromProjectUrl } from '../../src/utils.js';

describe('LambdaDB API Diagnostic', () => {
  it('should connect to LambdaDB and list collections', async () => {
    if (!process.env.LAMBDADB_PROJECT_API_KEY || !process.env.LAMBDADB_PROJECT_URL) {
      throw new Error('LAMBDADB_PROJECT_API_KEY and LAMBDADB_PROJECT_URL environment variables are required');
    }

    const serverURL = extractServerURLFromProjectUrl(process.env.LAMBDADB_PROJECT_URL!);
    const client = new LambdaDB({
      projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!,
      serverURL: serverURL,
      timeoutMs: 10000
    });
    console.log('🔍 Testing LambdaDB API connectivity...');
    console.log('📡 API Key:', process.env.LAMBDADB_PROJECT_API_KEY?.slice(0, 10) + '...');
    console.log('🌐 Project URL:', process.env.LAMBDADB_PROJECT_URL);
    console.log('🌐 Extracted Server URL:', serverURL);

    try {
      const response = await client.collections.list();
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
    if (!process.env.LAMBDADB_PROJECT_API_KEY || !process.env.LAMBDADB_PROJECT_URL) {
      throw new Error('LAMBDADB_PROJECT_API_KEY and LAMBDADB_PROJECT_URL environment variables are required');
    }

    const serverURL = extractServerURLFromProjectUrl(process.env.LAMBDADB_PROJECT_URL!);
    const client = new LambdaDB({
      projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!,
      serverURL: serverURL,
      timeoutMs: 10000
    });

    const testCollectionName = `diagnostic_test_${Date.now()}`;
    
    try {
      console.log(`🔨 Attempting to create collection: ${testCollectionName}`);
      
      await client.collections.create({
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
        await client.collections.delete({
          collectionName: testCollectionName,
        });
        console.log('🧹 Collection cleanup succeeded!');
      } catch (cleanupError) {
        console.warn('⚠️ Collection cleanup failed:', cleanupError.message);
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