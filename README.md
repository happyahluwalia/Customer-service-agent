# Customer Service Agent with Intelligent Routing

A sophisticated RAG-based (Retrieval Augmented Generation) application that intelligently routes and answers customer queries using domain-specific knowledge bases and AI-powered assistance.

## Overview

This application provides an intelligent customer service solution that:
- Automatically routes queries to the appropriate knowledge base (Products, Support, or Finance)
- Processes and indexes PDF documents for each domain
- Uses advanced RAG techniques for accurate information retrieval
- Falls back to web search when needed
- Maintains conversation history for context
- Provides a user-friendly interface built with Streamlit

## Features

### Core Capabilities
- 📚 Multiple Knowledge Bases
  - Product Information
  - Customer Support & FAQ
  - Financial Data
- 🤖 Intelligent Query Routing
- 📄 PDF Document Processing
- 💬 Contextual Responses
- 🔍 Web Search Fallback
- 📝 Conversation History

### Technical Highlights
- Vector similarity search for efficient routing
- LLM-based routing fallback
- Chunking and embedding of documents
- Hybrid search capabilities
- Real-time response generation

## Getting Started

### Prerequisites
```bash
python >= 3.8
pip install -r requirements.txt
```

### Required API Keys
- OpenAI API key
- Qdrant Cloud credentials (URL and API key)

### Installation
1. Clone the repository
```bash
git clone https://github.com/happyahluwalia/Customer-service-agent.git
cd customer-service-agent
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run customer_service_agent.py
```

## Technical Architecture

### High-Level Components

1. **User Interface Layer**
   - Built with Streamlit
   - Document upload interface
   - Query input and response display
   - Configuration management

2. **Routing Layer**
   - Vector similarity-based routing
   - LLM-based routing fallback
   - Confidence scoring system

3. **Knowledge Base Layer**
   - Document processing pipeline
   - Vector storage (Qdrant)
   - Embedding generation

4. **Response Generation Layer**
   - RAG-based answer generation
   - Web search fallback
   - Context integration

### Detailed Component Breakdown

#### 1. Document Processing Pipeline
- **PDF Loading**: Uses PyPDFLoader for document ingestion
- **Text Chunking**: 
  - Chunk size: 1000 characters
  - Overlap: 200 characters
- **Embedding Generation**: OpenAI text-embedding-3-small model
- **Vector Storage**: Qdrant collections

#### 2. Query Routing System
- **Primary Route**: Vector similarity search
  - Searches across all collections
  - Compares relevance scores
  - Uses confidence threshold (0.5)
- **Fallback Route**: LLM-based classification
  - Uses GPT-4 for classification
  - Strict output formatting
  - Domain-specific routing rules

#### 3. Answer Generation
- **RAG Implementation**:
  - Retrieves top 4 most relevant documents
  - Uses ChatGPT for answer generation
  - Maintains conversation context
- **Web Search Fallback**:
  - DuckDuckGo integration
  - Result summarization
  - Answer formatting

#### 4. Vector Store Architecture
- **Collections**:
  - products_collection
  - support_collection
  - finance_collection
- **Vector Configuration**:
  - Dimension: 1536 (OpenAI embeddings)
  - Distance metric: Cosine similarity

## Data Flow

1. **Document Ingestion**
   ```
   PDF → Chunks → Embeddings → Vector Store
   ```

2. **Query Processing**
   ```
   Query → Embedding → Similarity Search → Route Selection → Knowledge Base
   ```

3. **Answer Generation**
   ```
   Context + Query → LLM Processing → Formatted Response
   ```

## Configuration

### Environment Variables
```python
OPENAI_API_KEY=your_openai_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

### System Parameters
```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
SIMILARITY_K = 3
CONFIDENCE_THRESHOLD = 0.5
```

## Performance Considerations

### Optimization Points
- Document chunk size and overlap
- Number of documents retrieved (k)
- Confidence threshold for routing
- Vector similarity parameters

### Scalability Factors
- Vector store capacity
- API rate limits
- Document processing time
- Response generation speed

## Future Improvements

1. **Enhanced Routing**
   - Multi-label classification
   - Learning from user feedback
   - Dynamic confidence thresholds

2. **Advanced RAG**
   - Hybrid search implementation
   - Re-ranking of retrieved documents
   - Dynamic context window

3. **User Experience**
   - Multi-user support
   - Session management
   - Analytics dashboard

4. **Infrastructure**
   - Caching layer
   - Response time optimization
   - Batch processing for documents

## Troubleshooting

### Common Issues
1. **Document Processing Fails**
   - Check PDF format compatibility
   - Verify file size limits
   - Review chunk size settings

2. **Routing Issues**
   - Verify vector store connectivity
   - Check confidence thresholds
   - Review database population

3. **API Connection Problems**
   - Validate API keys
   - Check network connectivity
   - Verify rate limits

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- OpenAI for language models and embeddings
- Qdrant for vector storage
- Streamlit for the user interface
- LangChain for the RAG framework

## Screenshots
<img width="1275" alt="image" src="https://github.com/user-attachments/assets/1b46a487-8ede-4e6e-9f24-f5ac361ee484" />
<img width="906" alt="image" src="https://github.com/user-attachments/assets/5319ec9e-a2e5-44db-be7a-8d8cfd4c3cef" />

