/**
 * Comprehensive Integration Tests for LambdaDB Vector Store
 *
 * LambdaDB behavior:
 * - Only fields listed in indexConfigs at collection creation are indexed; unlisted fields are not searchable/filterable.
 * - Default is eventual consistency; we set defaultConsistentRead: true so query/fetch see writes immediately (minimal delay needed).
 * - Collections are created/deleted via the LambdaDB client directly; the vector store assumes the collection already exists.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { LambdaDBClient } from '@functional-systems/lambdadb';
import { LambdaDBVectorStore } from '../../src/index.js';
import { Document } from '@langchain/core/documents';
import { EmbeddingsInterface } from '@langchain/core/embeddings';

const TEST_VECTOR_DIMENSIONS = 3;

/** Create a collection via the LambdaDB client and wait for ACTIVE. */
async function createTestCollection(
  client: LambdaDBClient,
  collectionName: string,
  options?: {
    indexConfig?: Record<string, unknown>;
    similarity?: string;
    vectorField?: string;
    textField?: string;
  }
): Promise<void> {
  const vectorField = options?.vectorField ?? 'vector';
  const textField = options?.textField ?? 'page_content';
  type IndexConfigs = Parameters<LambdaDBClient['createCollection']>[0]['indexConfigs'];
  const indexConfigs = {
    [vectorField]: {
      type: 'vector' as const,
      dimensions: TEST_VECTOR_DIMENSIONS,
      similarity: (options?.similarity ?? 'cosine') as 'cosine' | 'euclidean' | 'dot_product' | 'max_inner_product',
    },
    [textField]: { type: 'text' as const, analyzers: ['english'] as const },
    ...(options?.indexConfig || {}),
  } as IndexConfigs;
  await client.createCollection({
    collectionName,
    indexConfigs,
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

/** Delete a collection via the LambdaDB client (best-effort, ignores errors). */
async function deleteTestCollection(client: LambdaDBClient, collectionName: string): Promise<void> {
  try {
    await client.collection(collectionName).delete();
  } catch {
    // ignore
  }
}

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
  let client: LambdaDBClient;
  let embeddings: TestEmbeddings;
  let collectionName: string;

  beforeEach(() => {
    if (!process.env.LAMBDADB_PROJECT_API_KEY) {
      throw new Error('LAMBDADB_PROJECT_API_KEY environment variable is required');
    }

    client = new LambdaDBClient({
      projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!,
      baseUrl: process.env.LAMBDADB_BASE_URL,
      projectName: process.env.LAMBDADB_PROJECT_NAME,
      timeoutMs: 30000,
    });
    embeddings = new TestEmbeddings();
    collectionName = `test_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;

    vectorStore = new LambdaDBVectorStore(embeddings, {
      collection: client.collection(collectionName),
      defaultConsistentRead: true,
    });
  });

  afterEach(async () => {
    try {
      await deleteTestCollection(client, collectionName);
      console.log(`🧹 Cleaned up: ${collectionName}`);
    } catch (error) {
      console.warn(`⚠️ Cleanup failed: ${(error as Error).message}`);
    }
  });

  describe('Collection Management', () => {
    it('should work with collection created via client', async () => {
      await createTestCollection(client, collectionName);

      const info = await vectorStore.getCollectionInfo();
      expect(info.name).toBe(collectionName);
      expect(info.status).toBe('ACTIVE');
      expect(info.documentCount).toBe(0);
      expect(info.indexConfigs).toHaveProperty('vector');

      await deleteTestCollection(client, collectionName);
    }, 90000);
  });

  describe('Document Operations', () => {
    it('should add documents and perform similarity search', async () => {
      await createTestCollection(client, collectionName);

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
      
      // Test addDocuments and return value (ids)
      const ids = await vectorStore.addDocuments(documents);
      expect(ids).toBeDefined();
      expect(Array.isArray(ids)).toBe(true);
      expect((ids as string[]).length).toBe(documents.length);
      (ids as string[]).forEach((id) => expect(typeof id === 'string' && id.length > 0).toBe(true));

      // defaultConsistentRead: true so minimal delay; short wait for API propagation
      await new Promise((resolve) => setTimeout(resolve, 500));

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
      await createTestCollection(client, collectionName);

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
      
      await new Promise((resolve) => setTimeout(resolve, 500));

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
      const fromTextsCollectionName = `fromtexts_${Date.now()}`;
      await createTestCollection(client, fromTextsCollectionName);

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
          collection: client.collection(fromTextsCollectionName),
          defaultConsistentRead: true,
        }
      );

      expect(store).toBeInstanceOf(LambdaDBVectorStore);

      await new Promise((resolve) => setTimeout(resolve, 500));
      const results = await store.similaritySearch('React framework', 2);
      expect(results).toHaveLength(2);

      const contents = results.map(doc => doc.pageContent);
      const hasReactResult = contents.some(content =>
        content.includes('React') || content.includes('framework')
      );
      expect(hasReactResult).toBe(true);

      await deleteTestCollection(client, fromTextsCollectionName);
    }, 120000);

    it('should create vector store from documents', async () => {
      const fromDocsCollectionName = `fromdocs_${Date.now()}`;
      await createTestCollection(client, fromDocsCollectionName);

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
          collection: client.collection(fromDocsCollectionName),
          defaultConsistentRead: true,
        }
      );

      expect(store).toBeInstanceOf(LambdaDBVectorStore);

      await new Promise((resolve) => setTimeout(resolve, 500));
      const results = await store.similaritySearch('typing system', 1);
      expect(results).toHaveLength(1);
      expect(results[0].pageContent).toContain('TypeScript');
      expect(results[0].metadata.language).toBe('typescript');

      await deleteTestCollection(client, fromDocsCollectionName);
    }, 120000);
  });

  describe('Advanced Features', () => {
    it('should perform max marginal relevance search', async () => {
      await createTestCollection(client, collectionName);

      const documents = [
        new Document({ pageContent: 'Apple is a fruit', metadata: { category: 'food' } }),
        new Document({ pageContent: 'Apple Inc. makes iPhones', metadata: { category: 'technology' } }),
        new Document({ pageContent: 'Orange is also a fruit', metadata: { category: 'food' } }),
        new Document({ pageContent: 'Google makes Android phones', metadata: { category: 'technology' } }),
        new Document({ pageContent: 'Banana is yellow fruit', metadata: { category: 'food' } })
      ];
      
      await vectorStore.addDocuments(documents);
      await new Promise((resolve) => setTimeout(resolve, 500));

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
      const customCollectionName = `custom_${Date.now()}`;
      await createTestCollection(client, customCollectionName, {
        vectorField: 'custom_vector',
        textField: 'text_content',
      });

      const customStore = new LambdaDBVectorStore(embeddings, {
        collection: client.collection(customCollectionName),
        textField: 'text_content',
        vectorField: 'custom_vector',
      });

      const info = await customStore.getCollectionInfo();
      expect(info.indexConfigs).toHaveProperty('custom_vector');

      await deleteTestCollection(client, customCollectionName);
    }, 90000);

    it('should handle different similarity metrics', async () => {
      const euclideanCollectionName = `euclidean_${Date.now()}`;
      await createTestCollection(client, euclideanCollectionName, { similarity: 'euclidean' });

      const euclideanStore = new LambdaDBVectorStore(embeddings, {
        collection: client.collection(euclideanCollectionName),
      });

      const info = await euclideanStore.getCollectionInfo();
      expect(info.indexConfigs?.vector?.similarity).toBe('euclidean');

      await deleteTestCollection(client, euclideanCollectionName);
    }, 90000);
  });

  describe('Delete Operations', () => {
    it('should throw when delete() is called without params', async () => {
      await createTestCollection(client, collectionName);
      await expect(vectorStore.delete()).rejects.toThrow('delete() requires explicit params');
      await expect(vectorStore.delete({})).rejects.toThrow('delete() requires explicit params');
    }, 30000);

    it('should delete documents by ids', async () => {
      await createTestCollection(client, collectionName);
      const documents = [
        new Document({ pageContent: 'Keep this document', metadata: { tag: 'keep' } }),
        new Document({ pageContent: 'Remove this document', metadata: { tag: 'remove' } }),
        new Document({ pageContent: 'Another to remove', metadata: { tag: 'remove' } }),
      ];
      const ids = (await vectorStore.addDocuments(documents)) as string[];
      expect(ids).toHaveLength(3);

      await new Promise((r) => setTimeout(r, 500));

      await vectorStore.delete({ ids: [ids[1], ids[2]] });

      await new Promise((r) => setTimeout(r, 500));

      const results = await vectorStore.similaritySearch('document', 5);
      expect(results).toHaveLength(1);
      expect(results[0].pageContent).toContain('Keep this document');
    }, 90000);

    it('should delete all documents with deleteAll: true', async () => {
      await createTestCollection(client, collectionName);
      await vectorStore.addDocuments([
        new Document({ pageContent: 'Doc A' }),
        new Document({ pageContent: 'Doc B' }),
      ]);
      await new Promise((r) => setTimeout(r, 500));

      await vectorStore.delete({ deleteAll: true });
      await new Promise((r) => setTimeout(r, 500));

      const info = await vectorStore.getCollectionInfo();
      expect(info.documentCount).toBe(0);
    }, 90000);

    it('should delete documents by filter when field is in indexConfigs', async () => {
      // LambdaDB only indexes fields listed in indexConfigs at collection creation
      const filterCollectionName = `test_filter_del_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
      const filterStore = new LambdaDBVectorStore(embeddings, {
        collection: client.collection(filterCollectionName),
        defaultConsistentRead: true,
      });
      try {
        await createTestCollection(client, filterCollectionName, { indexConfig: { kind: { type: 'keyword' } } });
        const documents = [
          new Document({ pageContent: 'Apple fruit', metadata: { kind: 'fruit' } }),
          new Document({ pageContent: 'Carrot vegetable', metadata: { kind: 'vegetable' } }),
          new Document({ pageContent: 'Banana fruit', metadata: { kind: 'fruit' } }),
        ];
        await filterStore.addDocuments(documents);
        await new Promise((r) => setTimeout(r, 500));

        await filterStore.delete({ filter: { queryString: { query: 'kind:vegetable' } } });
        await new Promise((r) => setTimeout(r, 500));

        const results = await filterStore.similaritySearch('food', 5);
        expect(results).toHaveLength(2);
        expect(results.every((d) => d.metadata?.kind === 'fruit')).toBe(true);
      } finally {
        await deleteTestCollection(client, filterCollectionName);
      }
    }, 90000);
  });

  describe('Search with Filter', () => {
    it('should filter by metadata when field is in indexConfigs (query string)', async () => {
      const filterCollectionName = `test_filter_srch_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
      const filterStore = new LambdaDBVectorStore(embeddings, {
        collection: client.collection(filterCollectionName),
        defaultConsistentRead: true,
      });
      try {
        await createTestCollection(client, filterCollectionName, { indexConfig: { lang: { type: 'keyword' } } });
        const documents = [
          new Document({ pageContent: 'JavaScript for frontend', metadata: { lang: 'js' } }),
          new Document({ pageContent: 'Python for backend', metadata: { lang: 'py' } }),
          new Document({ pageContent: 'TypeScript for frontend', metadata: { lang: 'ts' } }),
        ];
        await filterStore.addDocuments(documents);
        await new Promise((r) => setTimeout(r, 500));

        const results = await filterStore.similaritySearch('programming', 3, 'lang:py');
        expect(Array.isArray(results)).toBe(true);
        results.forEach((d) => {
          expect(d).toBeInstanceOf(Document);
          expect(d.pageContent).toBeDefined();
        });
        expect(results.every((d) => d.metadata?.lang === 'py')).toBe(true);
      } finally {
        await deleteTestCollection(client, filterCollectionName);
      }
    }, 90000);

    it('should filter by metadata when field is in indexConfigs (filter object)', async () => {
      const filterCollectionName = `test_filter_obj_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
      const filterStore = new LambdaDBVectorStore(embeddings, {
        collection: client.collection(filterCollectionName),
        defaultConsistentRead: true,
      });
      try {
        await createTestCollection(client, filterCollectionName, { indexConfig: { color: { type: 'keyword' } } });
        const documents = [
          new Document({ pageContent: 'Red apple', metadata: { color: 'red' } }),
          new Document({ pageContent: 'Green apple', metadata: { color: 'green' } }),
          new Document({ pageContent: 'Blue berry', metadata: { color: 'blue' } }),
        ];
        await filterStore.addDocuments(documents);
        await new Promise((r) => setTimeout(r, 500));

        const results = await filterStore.similaritySearchWithScore('fruit', 3, {
          queryString: { query: 'color:red' },
        });
        expect(Array.isArray(results)).toBe(true);
        results.forEach(([doc, score]) => {
          expect(doc).toBeInstanceOf(Document);
          expect(typeof score).toBe('number');
        });
        expect(results.every(([doc]) => doc.metadata?.color === 'red')).toBe(true);
      } finally {
        await deleteTestCollection(client, filterCollectionName);
      }
    }, 90000);
  });

  describe('Error Handling', () => {
    it('should handle invalid vector dimensions', async () => {
      await createTestCollection(client, collectionName);

      const invalidVector = [0.1, 0.2]; // Wrong dimensions (2 instead of 3)
      const document = new Document({ pageContent: 'Test doc' });
      
      await expect(vectorStore.addVectors([invalidVector], [document]))
        .rejects.toThrow('Vector dimension mismatch');
    }, 90000);

    it('should handle mismatched vectors and documents', async () => {
      await createTestCollection(client, collectionName);

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
      await createTestCollection(client, collectionName);

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
      expect(duration).toBeLessThan(30000);

      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Test search across batch
      const searchResults = await vectorStore.similaritySearch('topic 2', 5);
      expect(searchResults).toHaveLength(5);
      
    }, 180000); // Longer timeout for batch operations
  });
});