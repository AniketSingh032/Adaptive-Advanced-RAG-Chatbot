"""
Graph State Management Module.

This module defines the state structure for the conversational AI pipeline,
extending LangGraph's MessagesState to include custom attributes for
routing and document management.

Example:
    from src.utils.state import GraphState
    
    state = GraphState(
        messages=[],
        category="general",
        relevant_docs=""
    )
"""

from langgraph.graph import MessagesState
from typing import Literal

class GraphState(MessagesState):
    """
    Custom state manager for the conversation processing pipeline.
    
    Extends MessagesState to track:
    - Message history from the base class
    - Query routing categories
    - Retrieved document content
    
    Attributes:
        category (Literal["retriever", "general"]): 
            Determines the processing path for queries:
            - "retriever": Requires document lookup
            - "general": Uses general knowledge
            
        relevant_docs (str): 
            Stores retrieved documentation content for context-aware
            response generation. Empty for general knowledge queries.
    """
    category : Literal["retriever", "general"] 
    relevant_docs: str