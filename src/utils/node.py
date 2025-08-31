from pydantic import BaseModel, Field
from typing import Literal, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank, DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.schema import AIMessage
from src.config.logger import logger
from src.utils.state import GraphState
from src.utils.llm import llm, retriever, embeddings

class Node:
    """A class handling various nodes in the conversation processing pipeline.
    
    This class implements different processing nodes that handle:
    - Query routing between retrieval and general knowledge
    - General knowledge response generation
    - Document retrieval and relevance processing
    - Context-aware answer generation
    
    Each node maintains state and can be chained together in a processing pipeline.
    """

    def router_node(self, state: GraphState):
        """Route user queries to appropriate processing nodes.
        
        Args:
            state (Dict[str, Any]): Current conversation state containing 'question'
            
        Returns:
            Dict[str, Any]: Updated state with routing 'category'
        """
        try:
            class RouteQuery(BaseModel):
                category: Literal["retriever", "general"] = Field(
                    ..., description="Route the query to general or vectorstore."
                )

            structured_llm_router = llm.with_structured_output(RouteQuery)

            system = """You are an intelligent routing system responsible for analyzing and directing user queries efficiently.

            OBJECTIVE:
            Determine whether a query requires document retrieval or can be answered with general knowledge.

            INSTRUCTIONS:
            1. Analyze the query for:
               - Specific technical details requiring documentation
               - General conceptual questions
               - Context from previous conversation

            2. Categorize queries as:
               - 'retriever': For questions about:
                  * Specific DSPy documentation
                  * Technical implementations
                  * Code examples
                  * API usage
                  * Framework specifics
               
               - 'general': For questions about:
                  * Basic Chit Chat

            EXAMPLES:
            - "How do I implement DSPy's Predict module?" → retriever
            - "What is RAG in general?" → general
            - "Show me DSPy code examples" → retriever
            - "Explain the concept of language models" → general
                ## CHAT HISTORY
                {chat_history}


            """
            route_prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", "Question: \n\n {question}")
            ])

            question_router = route_prompt | structured_llm_router
            
            response = question_router.invoke({
                "question": state["messages"][-1].content if state["messages"] else "",
                "chat_history": state["messages"][-4:] if state["messages"] else []
            })
            logger.info(f"Routed to category: {response.category}")
            return {"category": response.category}
        except Exception as e:
            logger.error(f"Error in router_node: {str(e)}")
            raise

    def general_answer_node(self, state: GraphState) -> Dict[str, Any]:
        """Generate responses for general queries.
        
        Args:
            state (Dict[str, Any]): Conversation state with question
            
        Returns:
            Dict[str, Any]: Updated state with generated response
        """
        logger.info("Processing general knowledge query")
        try:
            instructions = """You are a professional AI assistant.  
                Your role is to handle chit-chat and casual conversation in a polite, friendly, and professional tone.  

                ## GUIDELINES
                - Keep responses short, warm, and natural.  
                - Maintain professionalism while staying approachable.  
                - Acknowledge greetings or casual remarks politely.  
                - Avoid technical or factual answers — focus only on conversation flow.
                - If a question can be answered using the context from chat history, do answer the question.  
                - If asked about anything outside chit-chat (e.g., facts, news, technical questions), respond with:  
                "My capabilities are limited to chit-chat. I cannot answer that question."  
                
                ## INPUT
                Current query: {question}  
                Chat history: {chat_history}
            """
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", instructions),
                ("human", "Question: \n\n {question}")
            ])
            messages = prompt_template.invoke({
                "question": state["messages"][-1].content if state["messages"] else "",
                "chat_history": state["messages"][-4:] if state["messages"] else []
            })
            response = llm.invoke(messages)
            logger.info("Generated general knowledge response")
            ai_message = AIMessage(content=response.content)
            return {"messages": [ai_message]}
        except Exception as e:
            logger.error(f"Error in general_answer_node: {str(e)}")
            raise

    def relevant_docs_node(self, state: GraphState) -> Dict[str, Any]:
        """Retrieve and process relevant documents for the query.
        
        Args:
            state (Dict[str, Any]): State containing query
            
        Returns:
            Dict[str, Any]: Updated state with relevant documents
        """
        logger.info("Retrieving relevant documents")
        try:
            compressor = FlashrankRerank(top_n=10)
            redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings, similarity_threshold=0.95)
            compressor_pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, compressor])
            retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm, include_original=True)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor_pipeline, base_retriever=retriever_from_llm
            )
            docs = compression_retriever.invoke(state["messages"][-1].content if state["messages"] else "")
            state["relevant_docs"] = "\n\n".join(doc.page_content for doc in docs)
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return state
        except Exception as e:
            logger.error(f"Error in relevant_docs_node: {str(e)}")
            raise

    def answer_generation_node(self, state: GraphState) -> Dict[str, Any]:
        """Generate contextual answers using retrieved documents.
        
        Args:
            state (Dict[str, Any]): State with question and relevant docs
            
        Returns:
            Dict[str, Any]: Updated state with generated answer
        """
        logger.info("Generating answer from documents")
        try:
            instructions = """You are a specialized AI assistant for the **DSPy framework**.  
                Answer queries only when relevant DSPy documentation is provided in context.  

                ## RULES
                1. Use **only** the given context to answer.  
                2. If the question is not related to DSPy, or the context is insufficient, reply exactly with:  
                - "I do not have capabilities to answer this question."  
                OR  
                - "I do not get enough context to answer that question."  
                3. Never guess, assume, or generate content outside the context.  

                ## RESPONSE STYLE
                - Start with a clear, direct answer.  
                - Use **bold** for key terms.  
                - Use bullet points or numbered steps for clarity.  
                - Put code in ```code blocks``` if provided.  
                - Cite documentation sections from context when possible.  

            AVAILABLE CONTEXT:
            {context}

            CHAT HISTORY:
            {chat_history}

            CURRENT QUERY:
            {question}
            """
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", instructions),
                ("human", "Question: \n\n {question}")
            ])
            messages = prompt_template.invoke({
                "context": state["relevant_docs"],
                "question": state["messages"][-1].content if state["messages"] else "",
                "chat_history": state["messages"][-4:] if state["messages"] else []
            })
            response = llm.invoke(messages)
            ai_message = AIMessage(content=response.content)
            return {"messages": [ai_message]}
        except Exception as e:
            logger.error(f"Error in answer_generation_node: {str(e)}")
            raise
    
node = Node()