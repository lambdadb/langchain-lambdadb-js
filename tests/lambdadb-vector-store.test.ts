import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LambdaDBVectorStore } from '../src/lambdadb-vector-store.js';
import { LambdaDBConfig } from '../src/types.js';
import { Document } from '@langchain/core/documents';
import { EmbeddingsInterface } from '@langchain/core/embeddings';

// Mock LambdaDB client
vi.mock('@functional-systems/lambdadb', () => ({
  LambdaDB: vi.fn().mockImplementation(() => ({
    collections: {
      list: vi.fn().mockResolvedValue({ collections: [] }),
      create: vi.fn().mockResolvedValue({}),
      delete: vi.fn().mockResolvedValue({}),
      get: vi.fn().mockResolvedValue({
        collection: {
          collectionName: 'test-collection',
          collectionStatus: 'ACTIVE', // Mock collection as ACTIVE to prevent waiting
          numDocs: 0,
          indexConfigs: {
            vector: {
              type: 'vector',
              dimensions: 3,
              similarity: 'cosine'
            }
          }
        }
      }),
      query: vi.fn().mockResolvedValue({
        docs: [
          {
            collection: 'test-collection',
            score: 0.95,
            doc: {
              content: 'Test document content',
              metadata: { source: 'test' }
            }
          }
        ]
      }),
      docs: {
        upsert: vi.fn().mockResolvedValue({}),
      }
    }
  }))
}));

// Mock embeddings interface
const mockEmbeddings: EmbeddingsInterface = {
  embedDocuments: vi.fn().mockResolvedValue([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
  embedQuery: vi.fn().mockResolvedValue([0.1, 0.2, 0.3]),
};

describe('LambdaDBVectorStore', () => {
  let config: LambdaDBConfig;
  let vectorStore: LambdaDBVectorStore;

  beforeEach(() => {
    config = {
      projectApiKey: 'test-api-key',
      collectionName: 'test-collection',
      vectorDimensions: 3,
      similarityMetric: 'cosine',
    };

    vectorStore = new LambdaDBVectorStore(mockEmbeddings, config);
    vi.clearAllMocks();
  });

  describe('constructor', () => {
    it('should create a vector store instance', () => {
      expect(vectorStore).toBeInstanceOf(LambdaDBVectorStore);
      expect(vectorStore._vectorstoreType()).toBe('lambdadb');
    });

    it('should validate required configuration', () => {
      const invalidConfig = {
        projectApiKey: '',
        collectionName: 'test',
        vectorDimensions: 3,
      };

      expect(() => {
        new LambdaDBVectorStore(mockEmbeddings, invalidConfig as LambdaDBConfig);
      }).toThrow('projectApiKey is required');
    });
  });

  describe('addDocuments', () => {
    it('should add documents to the vector store', async () => {
      const documents = [
        new Document({ pageContent: 'Document 1', metadata: { id: '1' } }),
        new Document({ pageContent: 'Document 2', metadata: { id: '2' } }),
      ];

      await vectorStore.addDocuments(documents);

      expect(mockEmbeddings.embedDocuments).toHaveBeenCalledWith(['Document 1', 'Document 2']);
      expect(vectorStore['client'].collections.docs.upsert).toHaveBeenCalled();
    }, 10000);

    it('should handle empty document array', async () => {
      await vectorStore.addDocuments([]);
      expect(mockEmbeddings.embedDocuments).not.toHaveBeenCalled();
    });
  });

  describe('addVectors', () => {
    it('should add vectors with documents', async () => {
      const vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
      const documents = [
        new Document({ pageContent: 'Document 1', metadata: { id: '1' } }),
        new Document({ pageContent: 'Document 2', metadata: { id: '2' } }),
      ];

      await vectorStore.addVectors(vectors, documents);

      expect(vectorStore['client'].collections.docs.upsert).toHaveBeenCalledWith({
        collectionName: 'test-collection',
        requestBody: {
          docs: expect.arrayContaining([
            expect.objectContaining({
              content: 'Document 1',
              vector: [0.1, 0.2, 0.3], // Updated to use 'vector' field
              id: '1' // metadata is now spread into the document  
            })
          ])
        }
      });
    }, 10000);

    it('should throw error if vectors and documents length mismatch', async () => {
      const vectors = [[0.1, 0.2, 0.3]];
      const documents = [
        new Document({ pageContent: 'Document 1' }),
        new Document({ pageContent: 'Document 2' }),
      ];

      await expect(vectorStore.addVectors(vectors, documents)).rejects.toThrow(
        'Vectors and documents length mismatch'
      );
    });
  });

  describe('similaritySearchVectorWithScore', () => {
    it('should perform similarity search', async () => {
      const queryVector = [0.1, 0.2, 0.3];
      const k = 5;

      const results = await vectorStore.similaritySearchVectorWithScore(queryVector, k);

      expect(vectorStore['client'].collections.query).toHaveBeenCalledWith({
        collectionName: 'test-collection',
        requestBody: {
          size: k,
          query: {
            knn: {
              field: "vector", // Updated to use 'vector' field
              queryVector: queryVector,
              k: k
            }
          },
          consistentRead: true, // Updated default value
        }
      });

      expect(results).toHaveLength(1);
      expect(results[0][0]).toBeInstanceOf(Document);
      expect(results[0][1]).toBe(0.95);
    });

    it('should validate vector dimensions', async () => {
      const invalidQueryVector = [0.1, 0.2]; // Wrong dimensions

      await expect(
        vectorStore.similaritySearchVectorWithScore(invalidQueryVector, 5)
      ).rejects.toThrow('Vector dimension mismatch: expected 3, got 2');
    });
  });

  describe('similaritySearch', () => {
    it('should perform similarity search with query text', async () => {
      const queryText = 'test query';
      const k = 3;

      const results = await vectorStore.similaritySearch(queryText, k);

      expect(mockEmbeddings.embedQuery).toHaveBeenCalledWith(queryText);
      expect(results).toHaveLength(1);
      expect(results[0]).toBeInstanceOf(Document);
    });
  });

  describe('static factory methods', () => {
    it('should create vector store from texts', async () => {
      const texts = ['Text 1', 'Text 2'];
      const metadatas = [{ id: '1' }, { id: '2' }];

      const store = await LambdaDBVectorStore.fromTexts(
        texts,
        metadatas,
        mockEmbeddings,
        config
      );

      expect(store).toBeInstanceOf(LambdaDBVectorStore);
      expect(mockEmbeddings.embedDocuments).toHaveBeenCalledWith(texts);
    }, 10000);

    it('should create vector store from documents', async () => {
      const documents = [
        new Document({ pageContent: 'Document 1', metadata: { id: '1' } }),
        new Document({ pageContent: 'Document 2', metadata: { id: '2' } }),
      ];

      const store = await LambdaDBVectorStore.fromDocuments(
        documents,
        mockEmbeddings,
        config
      );

      expect(store).toBeInstanceOf(LambdaDBVectorStore);
      expect(mockEmbeddings.embedDocuments).toHaveBeenCalled();
    }, 10000);
  });

  describe('maxMarginalRelevanceSearch', () => {
    it('should perform MMR search with proper parameters', async () => {
      const query = 'test query';
      const options = {
        k: 3,
        fetchK: 5,
        lambda: 0.7
      };

      const results = await vectorStore.maxMarginalRelevanceSearch(query, options);

      expect(mockEmbeddings.embedQuery).toHaveBeenCalledWith(query);
      expect(results).toHaveLength(1);
      expect(results[0]).toBeInstanceOf(Document);
    }, 10000);

    it('should handle MMR search with default parameters', async () => {
      const query = 'test query';
      const options = { k: 2 };

      const results = await vectorStore.maxMarginalRelevanceSearch(query, options);

      expect(mockEmbeddings.embedQuery).toHaveBeenCalledWith(query);
      expect(results).toHaveLength(1);
    }, 10000);

    it('should handle empty MMR results', async () => {
      // Mock empty response
      vectorStore['client'].collections.query = vi.fn().mockResolvedValue({
        docs: []
      });

      const query = 'test query';
      const options = { k: 3 };

      const results = await vectorStore.maxMarginalRelevanceSearch(query, options);

      expect(results).toHaveLength(0);
    }, 10000);
  });

  describe('collection management', () => {
    it('should create collection with correct configuration', async () => {
      await vectorStore.createCollection();

      expect(vectorStore['client'].collections.create).toHaveBeenCalledWith({
        collectionName: 'test-collection',
        indexConfigs: {
          vector: {  // Updated to use 'vector' field
            type: 'vector',
            dimensions: 3,
            similarity: 'cosine',
          }
        }
      });
    }, 10000);

    it('should delete collection', async () => {
      await vectorStore.deleteCollection();

      expect(vectorStore['client'].collections.delete).toHaveBeenCalledWith({
        collectionName: 'test-collection'
      });
    });
  });
});