# LangChain LambdaDB Integration

A production-ready TypeScript library that integrates [LambdaDB](https://lambdadb.ai) vector database with [LangChain.js](https://js.langchain.com/), providing seamless vector storage and retrieval capabilities for AI applications.

## Features

- 🚀 **Easy Integration**: Drop-in replacement for other LangChain vector stores
- 🎯 **Vector Similarity Search**: Support for cosine, euclidean, and dot product similarity metrics
- 🧠 **Max Marginal Relevance (MMR)**: Diverse search results balancing relevance and diversity
- 📊 **Batch Operations**: Efficient bulk document insertion and processing
- 🔍 **Flexible Configuration**: Custom field names, similarity metrics, and collection settings
- 🛡️ **Type Safety**: Full TypeScript support with comprehensive type definitions
- ⚡ **High Performance**: Leverages LambdaDB's optimized vector search engine; optional consistent reads when you need to see writes immediately
- 🧪 **Production Ready**: Comprehensive test suite (unit and integration tests)
- 🔄 **Retries**: Rely on the LambdaDB client for retries; vector store does not add its own retry layer
- 📦 **Collection lifecycle**: Create/delete collections via the LambdaDB client; vector store assumes the collection already exists
- 🗑️ **Document Deletion**: LangChain `delete()` support with server-side LambdaDB filter (by ids, filter, or deleteAll)

## Installation

```bash
npm install langchain-lambdadb @langchain/core
```

## Quick Start

```typescript
import { LambdaDBClient } from '@functional-systems/lambdadb';
import { LambdaDBVectorStore } from 'langchain-lambdadb';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';

// Create LambdaDB client once (reuse across multiple collections)
const client = new LambdaDBClient({
  projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!,
  baseUrl: process.env.LAMBDADB_BASE_URL ?? 'https://api.lambdadb.ai',
  projectName: process.env.LAMBDADB_PROJECT_NAME ?? 'your-project',
  timeoutMs: 30000,
});

// Initialize embeddings
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY
});

// Create the collection via the LambdaDB client first (vector store assumes it exists)
await client.createCollection({
  collectionName: 'my-documents',
  indexConfigs: {
    vector: { type: 'vector', dimensions: 1536, similarity: 'cosine' }, // dimensions must match your embeddings
    page_content: { type: 'text', analyzers: ['english'] },
  },
});

// Create vector store with a collection handle (embeddings, config) — same signature as base VectorStore
const vectorStore = new LambdaDBVectorStore(embeddings, {
  collection: client.collection('my-documents'),
});

// Add documents
const documents = [
  new Document({ 
    pageContent: 'LangChain is a framework for developing applications powered by language models.',
    metadata: { source: 'documentation', category: 'framework' }
  }),
  new Document({ 
    pageContent: 'LambdaDB is a vector database optimized for AI applications.',
    metadata: { source: 'documentation', category: 'database' }
  })
];

await vectorStore.addDocuments(documents);

// Perform similarity search
const results = await vectorStore.similaritySearch('What is LangChain?', 5);
console.log(results);
```

Using the same client for multiple collections:

```typescript
const storeA = new LambdaDBVectorStore(embeddings, { collection: client.collection('collection-a') });
const storeB = new LambdaDBVectorStore(embeddings, { collection: client.collection('collection-b') });
```

## Configuration Options

Connection (API key, base URL, project name) is set on **LambdaDBClient** from `@functional-systems/lambdadb`. The vector store assumes the **collection already exists**. If you call any operation (e.g. `addDocuments`, `similaritySearch`) when the collection does not exist, the store throws a clear error: *"Collection does not exist. Create it first using the LambdaDB client (...), then try again."* (The collection name is included in the message when available from the SDK.) Create and delete collections using the LambdaDB client directly (e.g. `client.createCollection(...)`, `client.collection(name).delete()`). Vector dimension is derived from the embeddings instance (cached on first use).

### LambdaDBVectorStoreConfig

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `collection` | `LambdaDBCollectionHandle` | ✅ | Collection handle (e.g. `client.collection('my-docs')`). |
| `textField` | `string` | ❌ | Field name for document content (default: 'page_content') |
| `vectorField` | `string` | ❌ | Field name for vectors (default: 'vector') |
| `defaultConsistentRead` | `boolean` | ❌ | Use consistent reads for query/fetch (default: false). Set true when you need to see writes immediately; otherwise LambdaDB uses eventual consistency. You can also override per call via search method options (e.g. `similaritySearch(..., { consistentRead: true })`). |

### Collection lifecycle

Create and delete collections via the **LambdaDB client**, not the vector store:

```typescript
// Create a collection (dimensions must match your embedding model)
await client.createCollection({
  collectionName: 'my-documents',
  indexConfigs: {
    vector: { type: 'vector', dimensions: 1536, similarity: 'cosine' },
    page_content: { type: 'text', analyzers: ['english'] },
    // Optional: keyword fields for filtering
    category: { type: 'keyword' },
  },
});

// Delete when no longer needed
await client.collection('my-documents').delete();
```

Similarity metrics: `'cosine'` (default), `'euclidean'`, `'dot_product'`, `'max_inner_product'`. See [LambdaDB docs](https://docs.lambdadb.ai) for `indexConfigs` and `partitionConfig`.

## Usage Examples

### Basic Vector Search

```typescript
import { LambdaDBClient } from '@functional-systems/lambdadb';
import { LambdaDBVectorStore } from 'langchain-lambdadb';
import { OpenAIEmbeddings } from '@langchain/openai';

const client = new LambdaDBClient({
  projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!,
  baseUrl: process.env.LAMBDADB_BASE_URL ?? 'https://api.lambdadb.ai',
  projectName: process.env.LAMBDADB_PROJECT_NAME ?? 'your-project',
});

const vectorStore = new LambdaDBVectorStore(new OpenAIEmbeddings(), { collection: client.collection('documents') });

// Search with custom parameters
const results = await vectorStore.similaritySearchWithScore('query text', 10);
results.forEach(([doc, score]) => {
  console.log(`Score: ${score}, Content: ${doc.pageContent}`);
});
```

### Using with Different Embedding Models

```typescript
import { LambdaDBClient } from '@functional-systems/lambdadb';
import { HuggingFaceTransformersEmbeddings } from '@langchain/community/embeddings/hf_transformers';

const client = new LambdaDBClient({
  projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!,
  baseUrl: process.env.LAMBDADB_BASE_URL ?? 'https://api.lambdadb.ai',
  projectName: process.env.LAMBDADB_PROJECT_NAME ?? 'your-project',
});

const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: 'Xenova/all-MiniLM-L6-v2',
});

const vectorStore = new LambdaDBVectorStore(embeddings, { collection: client.collection('hf-documents') });
```

### Creating from Texts and Metadata

```typescript
// Create vector store from texts (client in config)
const vectorStore = await LambdaDBVectorStore.fromTexts(
  texts,
  metadatas,
  embeddings,
  { collection: client.collection('my-collection') }
);
```

### Max Marginal Relevance (MMR) Search

```typescript
// MMR search for diverse results
const mmrResults = await vectorStore.maxMarginalRelevanceSearch(
  'machine learning frameworks', 
  {
    k: 5,        // Number of results to return
    fetchK: 20,  // Number of initial candidates to fetch
    lambda: 0.7  // Balance between relevance (1.0) and diversity (0.0)
  }
);
```

### Advanced Filtering

**Search** supports server-side filters (LambdaDB syntax) or a client-side function. Prefer server-side for efficiency.

```typescript
// Server-side: LambdaDB query string (recommended)
const results = await vectorStore.similaritySearchVectorWithScore(
  queryVector,
  5,
  'category:technology'
);

// Server-side: full LambdaDB filter object
const results2 = await vectorStore.similaritySearchVectorWithScore(queryVector, 5, {
  queryString: { query: 'category:technology AND year:2024' },
});

// Client-side: filter function (applied after fetch)
const filterFn = (doc: Document) => doc.metadata?.category === 'technology';
const results3 = await vectorStore.similaritySearchVectorWithScore(queryVector, 5, filterFn);
```

See [LambdaDB Query string](https://docs.lambdadb.ai/guides/search/query-string) for filter syntax.

### Deleting Documents

The store implements the LangChain VectorStore `delete()` interface. **You must pass explicit parameters** (no default to delete all, to avoid accidental wipe).

**By IDs** (most efficient when you know the ids):

```typescript
await vectorStore.delete({ ids: ['id1', 'id2'] });
```

**By LambdaDB filter** (recommended when filtering by metadata; server-side, one API call):

```typescript
// Query string – converted to LambdaDB queryString filter
await vectorStore.delete({ filter: 'genre:documentary AND year:2019' });

// Or full LambdaDB filter object
await vectorStore.delete({
  filter: { queryString: { query: 'genre:documentary AND year:2019' } },
});
```

See [LambdaDB Delete data](https://docs.lambdadb.ai/guides/documents/delete-data) and [Query string](https://docs.lambdadb.ai/guides/search/query-string) for filter syntax.

**Delete all documents** in the collection (explicit):

```typescript
await vectorStore.delete({ deleteAll: true });
```

**By client-side filter function** (fetches all docs then deletes by ids; use only when LambdaDB filter is not enough):

```typescript
await vectorStore.delete({
  filter: (doc) => doc.metadata.source === 'legacy',
});
```

### RAG (Retrieval-Augmented Generation) Integration

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const llm = new ChatOpenAI();
const retriever = vectorStore.asRetriever({
  searchType: 'similarity',
  searchKwargs: { k: 6 }
});

const chain = ConversationalRetrievalQAChain.fromLLM(llm, retriever);

const response = await chain.call({
  question: 'What is the main topic of the documents?',
  chat_history: []
});
```

## API Reference

### LambdaDBVectorStore Class

#### Constructor

```typescript
new LambdaDBVectorStore(embeddings: EmbeddingsInterface, config: LambdaDBVectorStoreConfig)
```

- Same signature as base `VectorStore`. Pass a collection handle via `config.collection` (e.g. `client.collection('my-docs')`).
- `config`: Must include `collection`; vector dimension is derived from embeddings (cached on first use).

#### Methods

##### `addDocuments(documents: Document[]): Promise<string[] \| void>`
Adds documents to the vector store with automatic embedding generation. Returns assigned document IDs.

##### `addVectors(vectors: number[][], documents: Document[]): Promise<string[] \| void>`
Adds pre-computed vectors with associated documents. Returns assigned document IDs. Payload ≤1MB uses a single `upsert`; larger payloads use `bulkUpsertDocs` (one call).

##### `similaritySearch(query: string, k?: number, filter?, _callbacks?, options?: VectorSearchOptions): Promise<Document[]>`
Performs similarity search with a text query. **options**: e.g. `{ consistentRead: true }` to override default consistency for this call.

##### `similaritySearchVectorWithScore(query: number[], k: number, filter?, options?: VectorSearchOptions): Promise<[Document, number][]>`
Performs similarity search with a vector query, returns documents with similarity scores. **Filter**: string or LambdaDB object → server-side `knn.filter`; function → client-side filter after fetch. **options**: e.g. `{ consistentRead: true }` to override default consistency for this call.

##### `maxMarginalRelevanceSearch(query: string, options?: MaxMarginalRelevanceSearchOptions): Promise<Document[]>`
Performs MMR search using vector similarity: fetches candidates with `includeVectors: true` and balances relevance to the query with diversity among selected documents (cosine similarity). **options** may include `consistentRead?: boolean` to override default consistency for this call.

##### `getCollectionInfo(): Promise<CollectionInfo>`
Returns information about the collection including status and document count.

##### `delete(_params?: Record<string, any>): Promise<void>` (LangChain VectorStore interface)
Deletes documents. **Requires explicit params** (no default). Use one of:

- `{ ids: string[] }` – delete by document IDs
- `{ filter: string | LambdaDBFilterObject }` – server-side delete (recommended); string is used as `queryString.query`
- `{ filter: (doc: Document) => boolean }` – client-side filter (fetches all, then deletes by ids)
- `{ deleteAll: true }` – delete all documents in the collection

##### `deleteDocuments(options: DeleteOptions): Promise<void>`
Lower-level delete with the same options as `delete()`: `ids`, `filter` (string, LambdaDB object, or function), or `deleteAll: true`.

#### Static Factory Methods

##### `fromTexts(texts: string[], metadatas: object[] | object, embeddings: EmbeddingsInterface, config: LambdaDBVectorStoreConfig): Promise<LambdaDBVectorStore>`
Creates a vector store from an array of texts. Pass collection via `config.collection`.

##### `fromDocuments(docs: Document[], embeddings: EmbeddingsInterface, config: LambdaDBVectorStoreConfig): Promise<LambdaDBVectorStore>`
Creates a vector store from an array of documents. Pass collection via `config.collection`.

## Migration from previous versions (breaking change)

If you were using the old constructor that accepted connection options in config:

**Before (old API, no longer supported):**
```typescript
const vectorStore = new LambdaDBVectorStore(embeddings, {
  projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!, // was in config
  collectionName: 'my-docs',
  vectorDimensions: 1536,
});
```

**After:**
```typescript
import { LambdaDBClient } from '@functional-systems/lambdadb';

const client = new LambdaDBClient({
  projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!,
  baseUrl: process.env.LAMBDADB_BASE_URL ?? 'https://api.lambdadb.ai',
  projectName: process.env.LAMBDADB_PROJECT_NAME ?? 'your-project',
  timeoutMs: 30000,
});

const vectorStore = new LambdaDBVectorStore(embeddings, {
  collection: client.collection('my-docs'),
});
```

- Create a `LambdaDBClient` once with your API key and URL/project; get a collection handle with `client.collection('name')` and pass it via **`config.collection`**.
- Remove `projectApiKey`, `baseUrl`, `projectName`, `serverURL`, and `vectorDimensions` from the config; dimension is derived from the embeddings (cached on first use).
- Use the same client for multiple collections: `client.collection('a')`, `client.collection('b')`.

## Environment Variables

You can set your LambdaDB credentials using environment variables:

```bash
export LAMBDADB_PROJECT_API_KEY="your-project-api-key"
export LAMBDADB_BASE_URL="https://api.lambdadb.ai"
export LAMBDADB_PROJECT_NAME="your-project"
```

## Error Handling

The library provides comprehensive error handling:

```typescript
try {
  await vectorStore.addDocuments(documents);
} catch (error) {
  if (error.message.includes('LambdaDB Error')) {
    console.error('LambdaDB service error:', error.message);
  } else if (error.message.includes('Vector dimension mismatch')) {
    console.error('Embedding dimension error:', error.message);
  } else {
    console.error('Unexpected error:', error.message);
  }
}
```

## Development

### Running Tests

```bash
# Run all tests
npm test

# Run only unit tests
npm run test:unit

# Run only integration tests (requires LAMBDADB_PROJECT_API_KEY, LAMBDADB_BASE_URL, LAMBDADB_PROJECT_NAME)
npm run test:integration
```

**Integration Tests**: Set `LAMBDADB_PROJECT_API_KEY`, `LAMBDADB_BASE_URL`, and `LAMBDADB_PROJECT_NAME` to run integration tests against real LambdaDB service.

### Building

```bash
npm run build
```

### Linting

```bash
npm run lint
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Implementation Details

### Key Features Implemented

- **Consistency**: Default is eventual consistency; set `defaultConsistentRead: true` in config or pass `{ consistentRead: true }` in search options when you need to see writes immediately (e.g. right after addDocuments)
- **Collection lifecycle**: Create/delete collections via the LambdaDB client; ensure collection is ACTIVE before using the vector store
- **Error Handling**: Clear errors when the collection is missing; retries are handled by the LambdaDB client
- **Field Name Configuration**: Supports custom field names for text and vector data
- **Upsert strategy**: Payload ≤1MB uses a single `upsert`; larger payloads use a single `bulkUpsertDocs` call
- **MMR**: Vector-based MMR with `includeVectors: true` and cosine similarity for relevance/diversity balance
- **Client options**: Prefer `baseUrl` + `projectName`; `serverURL` supported but deprecated
- **Test Coverage**: Unit and integration tests covering core functionality and edge cases

### LambdaDB Integration Notes

- Uses KNN query format: `{ knn: { field, queryVector, k } }`
- Prefer `baseUrl` + `projectName`; use `serverURL` (exact name, not `serverUrl`) only if overriding full URL
- Optional consistent read: set `defaultConsistentRead: true` in config or pass `{ consistentRead: true }` to search methods for immediate reads after writes
- Create collections via the client; optional `partitionConfig` supported. Vector store assumes the collection already exists.
- **Delete**: Prefer server-side filter (`filter` as string or LambdaDB object) for efficiency; `deleteAll: true` uses LambdaDB filter `{ queryString: { query: "*:*" } }`. [Delete data](https://docs.lambdadb.ai/guides/documents/delete-data)

## Links

- [LambdaDB Documentation](https://docs.lambdadb.ai/)
- [LangChain.js Documentation](https://js.langchain.com/)
- [TypeScript Client GitHub](https://github.com/lambdadb/lambdadb-typescript-client)
- [Python Integration Reference](https://github.com/lambdadb/langchain-lambdadb)

## Support

If you encounter any issues or have questions:

1. Check the [GitHub Issues](../../issues)
2. Review the [LambdaDB Documentation](https://docs.lambdadb.ai/)
3. Join the [LangChain Discord](https://discord.gg/langchain)