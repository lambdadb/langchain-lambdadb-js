# LangChain LambdaDB Integration

A production-ready TypeScript library that integrates [LambdaDB](https://lambdadb.ai) vector database with [LangChain.js](https://js.langchain.com/), providing seamless vector storage and retrieval capabilities for AI applications.

## Features

- 🚀 **Easy Integration**: Drop-in replacement for other LangChain vector stores
- 🎯 **Vector Similarity Search**: Support for cosine, euclidean, and dot product similarity metrics
- 🧠 **Max Marginal Relevance (MMR)**: Diverse search results balancing relevance and diversity
- 📊 **Batch Operations**: Efficient bulk document insertion and processing
- 🔍 **Flexible Configuration**: Custom field names, similarity metrics, and collection settings
- 🛡️ **Type Safety**: Full TypeScript support with comprehensive type definitions
- ⚡ **High Performance**: Leverages LambdaDB's optimized vector search engine with consistent reads
- 🧪 **Production Ready**: Comprehensive test suite with 43 passing tests (16 unit + 27 integration)
- 🔄 **Retry Logic**: Built-in exponential backoff for robust error handling
- 📈 **Collection Management**: Full lifecycle management with state monitoring

## Installation

```bash
npm install langchain-lambdadb @langchain/core
```

## Quick Start

```typescript
import { LambdaDBVectorStore } from 'langchain-lambdadb';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';

// Initialize embeddings
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY
});

// Configure LambdaDB connection
const config = {
  projectApiKey: process.env.LAMBDADB_API_KEY!,
  serverURL: process.env.LAMBDADB_SERVER_URL, // Optional: custom server
  collectionName: 'my-documents',
  vectorDimensions: 1536, // OpenAI embedding dimensions
  similarityMetric: 'cosine',
  // Optional: Configure retry behavior
  retryOptions: {
    maxAttempts: 3,
    initialDelay: 500,
    maxDelay: 5000
  }
};

// Create vector store
const vectorStore = new LambdaDBVectorStore(embeddings, config);

// Create collection if it doesn't exist
await vectorStore.createCollection();

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

## Configuration Options

### LambdaDBConfig

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `projectApiKey` | `string` | ✅ | Your LambdaDB project API key |
| `collectionName` | `string` | ✅ | Name of the collection to use |
| `vectorDimensions` | `number` | ✅ | Vector dimensions for embeddings |
| `similarityMetric` | `SimilarityMetric` | ❌ | Similarity metric (default: 'cosine') |
| `serverURL` | `string` | ❌ | Custom LambdaDB server URL (note: serverURL not serverUrl) |
| `textField` | `string` | ❌ | Field name for document content (default: 'content') |
| `vectorField` | `string` | ❌ | Field name for vectors (default: 'vector') |
| `validateCollection` | `boolean` | ❌ | Validate collection before operations (default: false) |
| `defaultConsistentRead` | `boolean` | ❌ | Use consistent reads by default (default: true) |
| `retryOptions` | `RetryOptions` | ❌ | Configure retry behavior with exponential backoff |

### Similarity Metrics

- `'cosine'` - Cosine similarity (default, recommended for most use cases)
- `'euclidean'` - Euclidean distance 
- `'dot_product'` - Dot product similarity

## Usage Examples

### Basic Vector Search

```typescript
import { LambdaDBVectorStore } from 'langchain-lambdadb';
import { OpenAIEmbeddings } from '@langchain/openai';

const vectorStore = new LambdaDBVectorStore(
  new OpenAIEmbeddings(),
  {
    projectApiKey: process.env.LAMBDADB_API_KEY!,
    collectionName: 'documents',
    vectorDimensions: 1536,
  }
);

// Search with custom parameters
const results = await vectorStore.similaritySearchWithScore('query text', 10);
results.forEach(([doc, score]) => {
  console.log(`Score: ${score}, Content: ${doc.pageContent}`);
});
```

### Using with Different Embedding Models

```typescript
import { HuggingFaceTransformersEmbeddings } from '@langchain/community/embeddings/hf_transformers';

// Using Hugging Face embeddings
const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: 'Xenova/all-MiniLM-L6-v2',
});

const vectorStore = new LambdaDBVectorStore(embeddings, {
  projectApiKey: process.env.LAMBDADB_API_KEY!,
  collectionName: 'hf-documents',
  vectorDimensions: 384, // all-MiniLM-L6-v2 dimensions
  similarityMetric: 'cosine'
});
```

### Creating from Texts and Metadata

```typescript
// Create vector store from texts
const texts = [
  'The quick brown fox jumps over the lazy dog.',
  'Machine learning is a subset of artificial intelligence.',
  'Vector databases enable efficient similarity search.'
];

const metadatas = [
  { category: 'literature' },
  { category: 'technology' },
  { category: 'database' }
];

const vectorStore = await LambdaDBVectorStore.fromTexts(
  texts,
  metadatas,
  embeddings,
  config
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

```typescript
// Search with custom filter
const filterFunction = (doc: Document) => {
  return doc.metadata.category === 'technology';
};

const filteredResults = await vectorStore.similaritySearchVectorWithScore(
  queryVector,
  5,
  filterFunction
);
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
new LambdaDBVectorStore(embeddings: EmbeddingsInterface, config: LambdaDBConfig)
```

#### Methods

##### `addDocuments(documents: Document[]): Promise<void>`
Adds documents to the vector store with automatic embedding generation.

##### `addVectors(vectors: number[][], documents: Document[]): Promise<void>`
Adds pre-computed vectors with associated documents.

##### `similaritySearch(query: string, k?: number, filter?: DocumentFilter): Promise<Document[]>`
Performs similarity search with a text query.

##### `similaritySearchVectorWithScore(query: number[], k: number, filter?: DocumentFilter): Promise<[Document, number][]>`
Performs similarity search with a vector query, returns documents with similarity scores.

##### `maxMarginalRelevanceSearch(query: string, options?: MMRSearchOptions): Promise<Document[]>`
Performs Max Marginal Relevance search for diverse results balancing relevance and diversity.

##### `createCollection(options?: Partial<CreateCollectionOptions>): Promise<void>`
Creates a new collection in LambdaDB with proper state monitoring.

##### `deleteCollection(): Promise<void>`
Deletes the collection from LambdaDB.

##### `getCollectionInfo(): Promise<CollectionInfo>`
Returns information about the collection including status and document count.

#### Static Factory Methods

##### `fromTexts(texts: string[], metadatas: object[] | object, embeddings: EmbeddingsInterface, config: LambdaDBConfig): Promise<LambdaDBVectorStore>`
Creates a vector store from an array of texts.

##### `fromDocuments(docs: Document[], embeddings: EmbeddingsInterface, config: LambdaDBConfig): Promise<LambdaDBVectorStore>`
Creates a vector store from an array of documents.

## Environment Variables

You can set your LambdaDB credentials using environment variables:

```bash
export LAMBDADB_API_KEY="your-api-key-here"
export LAMBDADB_SERVER_URL="https://your-instance.lambdadb.ai"  # Optional
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

# Run only integration tests (requires LAMBDADB_API_KEY)
npm run test:integration
```

**Integration Tests**: Set `LAMBDADB_API_KEY` and optionally `LAMBDADB_SERVER_URL` to run integration tests against real LambdaDB service.

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

- **Eventual Consistency Handling**: Uses `consistentRead: true` by default for immediate consistency
- **Collection State Management**: Proper waiting for collection to become ACTIVE before operations
- **Error Handling**: Comprehensive error handling with retry logic and exponential backoff
- **Field Name Configuration**: Supports custom field names for text and vector data
- **Batch Processing**: Efficient bulk operations with proper error handling
- **Test Coverage**: 43 tests covering all functionality including edge cases

### LambdaDB Integration Notes

- Uses KNN query format: `{ knn: { field, queryVector, k } }`
- Requires exact parameter name `serverURL` (not `serverUrl`)
- Supports immediate consistency with `consistentRead: true`
- Collection creation includes state polling until ACTIVE

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