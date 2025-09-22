/**
 * Basic usage example for LangChain LambdaDB integration
 * 
 * This example demonstrates:
 * - Setting up the vector store
 * - Adding documents
 * - Performing similarity search
 * - Using with different embedding models
 */

import { LambdaDBVectorStore } from '../src/index.js';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';

async function basicExample() {
  // Initialize OpenAI embeddings
  const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY!,
    model: 'text-embedding-3-small', // 1536 dimensions
  });

  // Configure LambdaDB
  const config = {
    projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!,
    collectionName: 'langchain-example',
    vectorDimensions: 1536,
    similarityMetric: 'cosine' as const,
  };

  console.log('🚀 Creating LambdaDB vector store...');
  const vectorStore = new LambdaDBVectorStore(embeddings, config);

  // Sample documents
  const documents = [
    new Document({
      pageContent: 'LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, and memory.',
      metadata: { source: 'langchain-docs', category: 'framework' }
    }),
    new Document({
      pageContent: 'Vector databases store high-dimensional vectors and enable efficient similarity search. They are crucial for RAG applications.',
      metadata: { source: 'vector-db-guide', category: 'database' }
    }),
    new Document({
      pageContent: 'LambdaDB is a serverless vector database that scales automatically and provides fast similarity search capabilities.',
      metadata: { source: 'lambdadb-docs', category: 'database' }
    }),
    new Document({
      pageContent: 'Embeddings are numerical representations of text that capture semantic meaning in high-dimensional space.',
      metadata: { source: 'ml-textbook', category: 'machine-learning' }
    }),
    new Document({
      pageContent: 'Retrieval-Augmented Generation (RAG) combines pre-trained language models with external knowledge retrieval.',
      metadata: { source: 'rag-paper', category: 'machine-learning' }
    })
  ];

  console.log('📝 Adding documents to vector store...');
  await vectorStore.addDocuments(documents);
  console.log(`✅ Added ${documents.length} documents`);

  // Perform similarity searches
  console.log('\n🔍 Performing similarity searches...');

  // Search 1: Basic similarity search
  const query1 = 'What is a vector database?';
  console.log(`\nQuery: "${query1}"`);
  const results1 = await vectorStore.similaritySearch(query1, 3);
  results1.forEach((doc, idx) => {
    console.log(`${idx + 1}. ${doc.pageContent.substring(0, 100)}... (${doc.metadata.category})`);
  });

  // Search 2: Search with scores
  const query2 = 'How does LangChain work?';
  console.log(`\nQuery with scores: "${query2}"`);
  const results2 = await vectorStore.similaritySearchWithScore(query2, 2);
  results2.forEach(([doc, score], idx) => {
    console.log(`${idx + 1}. Score: ${score.toFixed(4)} | ${doc.pageContent.substring(0, 80)}... (${doc.metadata.source})`);
  });

  // Search 3: Search with filter
  console.log('\n🎯 Filtered search (machine-learning category only):');
  const filterFn = (doc: Document) => doc.metadata.category === 'machine-learning';
  const filteredResults = await vectorStore.similaritySearchVectorWithScore(
    await embeddings.embedQuery('machine learning concepts'),
    3,
    filterFn
  );
  
  filteredResults.forEach(([doc, score], idx) => {
    console.log(`${idx + 1}. Score: ${score.toFixed(4)} | ${doc.pageContent.substring(0, 80)}...`);
  });

  console.log('\n✨ Example completed successfully!');
}

async function fromTextsExample() {
  console.log('\n📚 Creating vector store from texts...');
  
  const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY!,
    model: 'text-embedding-3-small',
  });

  const texts = [
    'Artificial Intelligence is transforming how we interact with technology.',
    'Machine Learning algorithms can learn patterns from data automatically.',
    'Natural Language Processing enables computers to understand human language.',
    'Deep Learning uses neural networks with multiple layers for complex tasks.'
  ];

  const metadatas = [
    { topic: 'AI', difficulty: 'beginner' },
    { topic: 'ML', difficulty: 'intermediate' },
    { topic: 'NLP', difficulty: 'intermediate' },
    { topic: 'DL', difficulty: 'advanced' }
  ];

  const vectorStore = await LambdaDBVectorStore.fromTexts(
    texts,
    metadatas,
    embeddings,
    {
      projectApiKey: process.env.LAMBDADB_PROJECT_API_KEY!,
      collectionName: 'from-texts-example',
      vectorDimensions: 1536,
    }
  );

  const searchResults = await vectorStore.similaritySearch('neural networks', 2);
  console.log('📊 Search results:');
  searchResults.forEach((doc, idx) => {
    console.log(`${idx + 1}. ${doc.pageContent} (Topic: ${doc.metadata.topic})`);
  });
}

// Run examples if this file is executed directly
if (require.main === module) {
  (async () => {
    try {
      await basicExample();
      await fromTextsExample();
    } catch (error) {
      console.error('❌ Error running examples:', error);
      process.exit(1);
    }
  })();
}

export { basicExample, fromTextsExample };