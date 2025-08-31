"""
Language Model Service Module.

This module provides a centralized service for managing language model operations,
including embedding generation, vector storage, and document retrieval capabilities.
It handles initialization and configuration of various LLM components.

Example:
    from src.utils.llm import llm, retriever, embeddings
    
    response = llm.predict("What is RAG?")
    similar_docs = retriever.get_relevant_documents("query")
"""

from typing import Any
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from src.config.config import settings
from src.config.logger import logger

class LLMService:
    """
    Language Model Service Manager.

    Centralizes the management of:
    - Large Language Model (LLM) initialization and operations
    - Text embedding generation
    - Vector store operations
    - Document retrieval functionality

    Each component is initialized lazily upon first use to optimize resource usage.
    """

    def _initialize_llm(self) -> ChatGroq:
        """
        Initialize and configure the Groq Language Model instance.
        
        Creates a ChatGroq instance using configuration from settings:
        - Model identifier from GROQ_MODEL
        - Authentication from GROQ_API_KEY
        
        Returns:
            ChatGroq: Configured chat model ready for text generation
        
        Raises:
            RuntimeError: On API, authentication, or configuration failures
        """
        logger.info("Initializing Language Model...")
        try:
            llm = ChatGroq(
                model=settings.GROQ_MODEL, 
                api_key=settings.GROQ_API_KEY,
            
            )
            logger.info("Language Model initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize Language Model: {str(e)}")
            raise RuntimeError(f"Error initializing Language Model: {e}")

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Set up the text embedding model using HuggingFace.
        
        Configures an embedding model that:
        - Converts text to vector representations
        - Uses the model specified in EMBEDDINGS_MODEL setting
        - Supports document similarity operations
        
        Returns:
            HuggingFaceEmbeddings: Ready-to-use embedding model
        
        Raises:
            RuntimeError: If model loading or initialization fails
        """
        logger.info(f"Initializing embeddings with model: {settings.EMBEDDINGS_MODEL}")
        try:
            embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL)
            logger.info("Embeddings model initialized successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise RuntimeError(f"Error initializing embeddings: {e}")

    def _initialize_vectorstore(self) -> Chroma:
        """
        Configure the Chroma vector database for document storage.
        
        Creates a persistent vector store that:
        - Integrates with the configured embedding model
        - Maintains persistence in PERSIST_DIRECTORY
        - Provides similarity search capabilities
        
        Returns:
            Chroma: Configured vector store instance
        
        Raises:
            RuntimeError: On storage initialization or embedding setup failures
        """
        logger.info("Initializing vector store...")
        try:
            vectordb = Chroma(
                persist_directory=settings.PERSIST_DIRECTORY, 
                embedding_function=self._initialize_embeddings()
            )
            logger.info("Vector store initialized successfully")
            return vectordb
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise RuntimeError(f"Error initializing vector store: {e}")

    def get_retriever(self) -> Any:
        """
        Create a document retriever for similarity search operations.
        
        Configures a retriever with:
        - Cosine similarity search
        - Top-10 document retrieval
        - Integration with the vector store
        
        Returns:
            BaseRetriever: Configured similarity search retriever
        
        Raises:
            RuntimeError: On retriever or vector store initialization failures
        
        Example:
            retriever = llm_service.get_retriever()
            docs = retriever.get_relevant_documents("query")
        """
        logger.info("Setting up retriever...")
        try:
            vectordb = self._initialize_vectorstore()
            retriever = vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
            logger.info("Retriever setup completed successfully")
            return retriever
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {str(e)}")
            raise RuntimeError(f"Error initializing retriever: {e}")

llm = LLMService()._initialize_llm()
retriever = LLMService().get_retriever()
embeddings = LLMService()._initialize_embeddings()