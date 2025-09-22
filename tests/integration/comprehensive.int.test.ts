/**
 * Comprehensive Integration Tests for LambdaDB Vector Store
 * 
 * Tests the complete LangChain integration functionality with real LambdaDB service.
 * Based on LangChain vector store test patterns and Python integration reference.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { LambdaDBVectorStore } from '../../src/index.js';
import { Document } from '@langchain/core/documents';
import { EmbeddingsInterface } from '@langchain/core/embeddings';

// Test embeddings with deterministic output for consistent testing
class TestEmbeddings implements EmbeddingsInterface {
  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map((text, idx) => this.createVector(text, idx));
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.createVector(text, 0);
  }

  private createVector(text: string, idx: number): number[] {
    // Create deterministic vectors based on text content and index
    const textHash = text.split('').reduce((hash, char) => hash + char.charCodeAt(0), 0);
    const baseValue = (textHash % 100) / 100;
    
    return [
      Math.sin(baseValue + idx) * 0.8,
      Math.cos(baseValue + idx) * 0.6, 
      Math.sin(baseValue * 2 + idx) * 0.4
    ];
  }
}

describe('LambdaDB Comprehensive Integration Tests', () => {
  let vectorStore: LambdaDBVectorStore;
  let embeddings: TestEmbeddings;
  let collectionName: string;

  beforeEach(() => {
    if (!process.env.LAMBDADB_API_KEY) {
      throw new Error('LAMBDADB_API_KEY environment variable is required');
    }

    embeddings = new TestEmbeddings();
    collectionName = `test_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;
    
    vectorStore = new LambdaDBVectorStore(embeddings, {
      projectApiKey: process.env.LAMBDADB_API_KEY!,
      ...(process.env.LAMBDADB_SERVER_URL && { serverURL: process.env.LAMBDADB_SERVER_URL }),
      collectionName,
      vectorDimensions: 3,
      similarityMetric: 'cosine',
      retryOptions: { maxAttempts: 2, initialDelay: 100 }, // Faster retries for tests
    });
  });

  afterEach(async () => {
    // Cleanup: Always try to delete the collection
    try {
      await vectorStore.deleteCollection();
      console.log(`🧹 Cleaned up: ${collectionName}`);
    } catch (error) {
      console.warn(`⚠️ Cleanup failed: ${error.message}`);
    }
  });

  describe('Collection Management', () => {
    it('should create and manage collection lifecycle', async () => {
      // Test collection creation
      await vectorStore.createCollection();
      
      // Verify collection info
      const info = await vectorStore.getCollectionInfo();
      expect(info.name).toBe(collectionName);
      expect(info.status).toBe('ACTIVE');
      expect(info.documentCount).toBe(0);
      expect(info.indexConfigs).toHaveProperty('vector'); // Updated to match new default
      
      // Test deletion
      await vectorStore.deleteCollection();
    }, 90000);
  });

  describe('Document Operations', () => {
    it('should add documents and perform similarity search', async () => {
      await vectorStore.createCollection();
      
      const documents = [
        new Document({ 
          pageContent: 'The quick brown fox jumps over the lazy dog',
          metadata: { id: 'doc1', category: 'animals' }
        }),
        new Document({ 
          pageContent: 'Python is a programming language',
          metadata: { id: 'doc2', category: 'programming' }
        }),
        new Document({ 
          pageContent: 'Machine learning uses algorithms to find patterns',
          metadata: { id: 'doc3', category: 'ai' }
        })
      ];
      
      // Test addDocuments
      await vectorStore.addDocuments(documents);
      
      // Wait a moment for indexing
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Test similarity search by text
      const searchResults = await vectorStore.similaritySearch('programming language', 2);
      expect(searchResults).toHaveLength(2);
      expect(searchResults[0]).toBeInstanceOf(Document);
      
      // Check that we get relevant results (flexible ranking)
      const contents = searchResults.map(doc => doc.pageContent);
      const hasProgrammingResult = contents.some(content => 
        content.includes('Python') || content.includes('programming')
      );
      expect(hasProgrammingResult).toBe(true);
      
      // Test similarity search with scores
      const scoredResults = await vectorStore.similaritySearchWithScore('programming', 3);
      expect(scoredResults).toHaveLength(3);
      expect(scoredResults[0]).toHaveLength(2); // [Document, score]
      expect(scoredResults[0][1]).toBeTypeOf('number');
      expect(scoredResults[0][1]).toBeGreaterThan(0);
    }, 120000);

    it('should handle vector operations directly', async () => {
      await vectorStore.createCollection();
      
      const vectors = [
        [0.8, 0.6, 0.2],
        [0.1, 0.9, 0.3], 
        [0.4, 0.5, 0.8]
      ];
      
      const documents = [
        new Document({ pageContent: 'First document', metadata: { id: 'v1' } }),
        new Document({ pageContent: 'Second document', metadata: { id: 'v2' } }),
        new Document({ pageContent: 'Third document', metadata: { id: 'v3' } })
      ];
      
      // Test addVectors
      await vectorStore.addVectors(vectors, documents);
      
      // Wait for indexing
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Test vector similarity search
      const queryVector = [0.7, 0.5, 0.1]; // Similar to first vector
      const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 2);
      
      expect(results).toHaveLength(2);
      expect(results[0][0]).toBeInstanceOf(Document);
      expect(results[0][1]).toBeTypeOf('number');
      
      // First result should be closest to our query vector
      expect(results[0][0].pageContent).toContain('First document');
    }, 120000);
  });

  describe('Factory Methods', () => {
    it('should create vector store from texts', async () => {
      const texts = [
        'JavaScript is a versatile programming language',
        'React is a popular frontend framework',
        'Node.js enables server-side JavaScript'
      ];
      
      const metadatas = [
        { id: 'js1', topic: 'javascript' },
        { id: 'js2', topic: 'react' },
        { id: 'js3', topic: 'nodejs' }
      ];
      
      const store = await LambdaDBVectorStore.fromTexts(
        texts,
        metadatas,
        embeddings,
        {
          projectApiKey: process.env.LAMBDADB_API_KEY!,
          ...(process.env.LAMBDADB_SERVER_URL && { serverURL: process.env.LAMBDADB_SERVER_URL }),
          collectionName: `fromtexts_${Date.now()}`,
          vectorDimensions: 3,
          similarityMetric: 'cosine',
        }
      );
      
      expect(store).toBeInstanceOf(LambdaDBVectorStore);
      
      // Test search functionality
      await new Promise(resolve => setTimeout(resolve, 2000));
      const results = await store.similaritySearch('React framework', 2);
      expect(results).toHaveLength(2);
      
      // Check that we get relevant results (flexible ranking)
      const contents = results.map(doc => doc.pageContent);
      const hasReactResult = contents.some(content => 
        content.includes('React') || content.includes('framework')
      );
      expect(hasReactResult).toBe(true);
      
      // Cleanup
      await store.deleteCollection();
    }, 120000);

    it('should create vector store from documents', async () => {
      const docs = [
        new Document({
          pageContent: 'TypeScript adds static typing to JavaScript',
          metadata: { language: 'typescript', difficulty: 'intermediate' }
        }),
        new Document({
          pageContent: 'Python is known for its simplicity and readability',
          metadata: { language: 'python', difficulty: 'beginner' }
        })
      ];
      
      const store = await LambdaDBVectorStore.fromDocuments(
        docs,
        embeddings,
        {
          projectApiKey: process.env.LAMBDADB_API_KEY!,
          ...(process.env.LAMBDADB_SERVER_URL && { serverURL: process.env.LAMBDADB_SERVER_URL }),
          collectionName: `fromdocs_${Date.now()}`,
          vectorDimensions: 3,
          similarityMetric: 'cosine',
        }
      );
      
      expect(store).toBeInstanceOf(LambdaDBVectorStore);
      
      // Test search functionality
      await new Promise(resolve => setTimeout(resolve, 2000));
      const results = await store.similaritySearch('typing system', 1);
      expect(results).toHaveLength(1);
      expect(results[0].pageContent).toContain('TypeScript');
      expect(results[0].metadata.language).toBe('typescript');
      
      // Cleanup
      await store.deleteCollection();
    }, 120000);
  });

  describe('Advanced Features', () => {
    it('should perform max marginal relevance search', async () => {
      await vectorStore.createCollection();
      
      const documents = [
        new Document({ pageContent: 'Apple is a fruit', metadata: { category: 'food' } }),
        new Document({ pageContent: 'Apple Inc. makes iPhones', metadata: { category: 'technology' } }),
        new Document({ pageContent: 'Orange is also a fruit', metadata: { category: 'food' } }),
        new Document({ pageContent: 'Google makes Android phones', metadata: { category: 'technology' } }),
        new Document({ pageContent: 'Banana is yellow fruit', metadata: { category: 'food' } })
      ];
      
      await vectorStore.addDocuments(documents);
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Test MMR search - should return diverse results
      const mmrResults = await vectorStore.maxMarginalRelevanceSearch('Apple fruit', {
        k: 3,
        fetchK: 5,
        lambda: 0.5 // Balance between relevance and diversity
      });
      
      expect(mmrResults).toHaveLength(3);
      expect(mmrResults[0]).toBeInstanceOf(Document);
      
      // Should include both Apple contexts (fruit and technology) for diversity
      const contents = mmrResults.map(doc => doc.pageContent);
      expect(contents.some(content => content.includes('fruit'))).toBe(true);
    }, 120000);
  });

  describe('Configuration Options', () => {
    it('should work with custom field names', async () => {
      const customStore = new LambdaDBVectorStore(embeddings, {
        projectApiKey: process.env.LAMBDADB_API_KEY!,
        ...(process.env.LAMBDADB_SERVER_URL && { serverURL: process.env.LAMBDADB_SERVER_URL }),
        collectionName: `custom_${Date.now()}`,
        vectorDimensions: 3,
        similarityMetric: 'cosine',
        textField: 'text_content',
        vectorField: 'custom_vector',
      });
      
      try {
        await customStore.createCollection();
        
        const info = await customStore.getCollectionInfo();
        expect(info.indexConfigs).toHaveProperty('custom_vector');
        
        await customStore.deleteCollection();
      } catch (error) {
        await customStore.deleteCollection();
        throw error;
      }
    }, 90000);

    it('should handle different similarity metrics', async () => {
      const euclideanStore = new LambdaDBVectorStore(embeddings, {
        projectApiKey: process.env.LAMBDADB_API_KEY!,
        ...(process.env.LAMBDADB_SERVER_URL && { serverURL: process.env.LAMBDADB_SERVER_URL }),
        collectionName: `euclidean_${Date.now()}`,
        vectorDimensions: 3,
        similarityMetric: 'euclidean',
      });
      
      try {
        await euclideanStore.createCollection();
        
        const info = await euclideanStore.getCollectionInfo();
        expect(info.indexConfigs?.vector.similarity).toBe('euclidean'); // Updated field name
        
        await euclideanStore.deleteCollection();
      } catch (error) {
        await euclideanStore.deleteCollection();
        throw error;
      }
    }, 90000);
  });

  describe('Error Handling', () => {
    it('should handle invalid vector dimensions', async () => {
      await vectorStore.createCollection();
      
      const invalidVector = [0.1, 0.2]; // Wrong dimensions (2 instead of 3)
      const document = new Document({ pageContent: 'Test doc' });
      
      await expect(vectorStore.addVectors([invalidVector], [document]))
        .rejects.toThrow('Vector dimension mismatch');
    }, 90000);

    it('should handle mismatched vectors and documents', async () => {
      await vectorStore.createCollection();
      
      const vectors = [[0.1, 0.2, 0.3]];
      const documents = [
        new Document({ pageContent: 'Doc 1' }),
        new Document({ pageContent: 'Doc 2' }) // Extra document
      ];
      
      await expect(vectorStore.addVectors(vectors, documents))
        .rejects.toThrow('Vectors and documents length mismatch');
    }, 90000);
  });

  describe('Performance and Scale', () => {
    it('should handle batch operations efficiently', async () => {
      await vectorStore.createCollection();
      
      // Create a larger batch of documents
      const batchSize = 50;
      const documents = Array.from({ length: batchSize }, (_, i) => 
        new Document({
          pageContent: `Document number ${i} with unique content about topic ${i % 5}`,
          metadata: { 
            index: i, 
            batch: Math.floor(i / 10),
            topic: `topic_${i % 5}`
          }
        })
      );
      
      const startTime = Date.now();
      await vectorStore.addDocuments(documents);
      const duration = Date.now() - startTime;
      
      console.log(`📊 Added ${batchSize} documents in ${duration}ms`);
      expect(duration).toBeLessThan(30000); // Should complete within 30 seconds
      
      // Wait for indexing
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Test search across batch
      const searchResults = await vectorStore.similaritySearch('topic 2', 5);
      expect(searchResults).toHaveLength(5);
      
    }, 180000); // Longer timeout for batch operations
  });
});