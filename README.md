## Adaptive+Advanced-RAG based Chatbot

### Overview

This project implements Adaptive + Advanced RAG (Retrieval-Augmented Generation) system using LangGraph and Langchain pre built components for intelligent document retrieval and query processing. The system combines adaptive query routing with advanced document processing techniques to deliver contextual, accurate responses while maintaining conversation continuity.

The system intelligently routes queries between general knowledge responses and document-based retrieval, optimizing both response quality and processing efficiency through advanced RAG techniques.

### Architecture

<p align="center">
  <img src="assets/Adaptive+Advanced-Architecture.png.png"/>
</p>

The system uses a **multi-node workflow architecture** powered by LangGraph with the following key components:

* **Router Node:** Intelligent LLM-based query classification that determines whether queries require document retrieval or can be answered with general knowledge
* **General Answer Node:** Handles conceptual and general knowledge queries using direct LLM responses
* **Document Retrieval Node:** Implements advanced document processing with multi-query retrieval, compression, and reranking
* **Answer Generation Node:** Generates contextual responses using retrieved documents with proper citation and formatting

**Workflow Process:**
1. User query enters the router node
2. LLM performs intelligent classification (general vs retrieval-based)  
3. Query routes to appropriate processing path
4. Advanced document processing (if retrieval path selected)
5. Generate contextual response with proper formatting
6. Maintain conversation memory across interactions

### Key Features

#### Advanced RAG Components
- **Multi-Query Retriever:** Generates multiple query variations for comprehensive document retrieval
- **FlashrankRerank:** State-of-the-art document reranking for relevance optimization  
- **Contextual Compression:** Intelligent document compression while preserving key information
- **Redundancy Filtering:** Removes duplicate or highly similar documents using embeddings similarity
- **Document Pipeline:** Chains multiple compression and filtering techniques

#### Adaptive Capabilities  
- **Intelligent Query Routing:** LLM-based classification with structured output
- **Dynamic Workflow Selection:** Automatic routing between retrieval and knowledge-based responses
- **Context-Aware Processing:** Maintains conversation history and adapts responses accordingly
- **Memory Management:** Persistent conversation state with thread-based memory

#### Technical Stack
- **LangGraph:** Workflow orchestration and state management
- **LangChain:** RAG components and retrieval systems
- **Groq:** High-performance LLM provider (ChatGroq)
- **HuggingFace:** Embeddings model for semantic search
- **Chroma:** Vector database for document storage and similarity search
- **Python 3.11+:** Core programming language

## Getting Started

### Prerequisites

* Requires Python version **3.11** or higher.

### Installing

* Clone the repository:
```
git clone https://github.com/AniketSingh032/Adaptive-Advanced-RAG-Chatbot.git
```
* Navigate to the root directory and create virtual environment, activate it and install Poetry:
```
virtualenv venv
venv/scripts/activate
pip install poetry
```
* Initialize and Install the Poetry dependencies:
```
poetry init
poetry install --no-root 

```
* Create a .env file in the root directory and add the environment variables:
```
GROQ_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=
EMBEDDINGs_MODEL = sentence-transformers/all-MiniLM-L6-v2
PERSIST_DIRECTORY = embeddings_db
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=your_langsmith_tracing
LANGSMITH_ENDPOINT=your_langsmith_endpoint
LANGSMITH_PROJECT=your_langsmith_project
```

### Running

* Run the LangGraph Studio using:
```
langgraph dev
```

## Limitations

⚠️ **Important Notice: This repository is for educational and tutorial purposes only.**

### Known Limitations

1. **Query Routing Accuracy:** There are instances where the system may incorrectly route queries between general knowledge and document retrieval paths. The LLM-based routing, while sophisticated, is not 100% accurate and may misclassify edge cases or ambiguous queries.

2. **Educational Purpose Only:** This repository is designed for **learning and tutorial purposes** and should **not be used directly in production environments** without significant modifications and enhancements.

3. **Code Optimization Opportunities:** There are numerous areas where the code quality, performance, and accuracy can be significantly improved, including:
   - Error handling and edge case management
   - Prompt engineering and optimization

4. **Production Readiness:** The current implementation lacks several production-level requirements such as:
   - Comprehensive testing suite
   - Scalability optimizations
   - Security hardening
   - Performance monitoring
   - Error recovery mechanisms
   - API rate limiting and throttling

### Future Improvements 😉

**Improvements and production-level code coming in future updates!** We're actively working on:
- Enhanced query routing accuracy
- Performance optimizations
- Production-ready deployment configurations
