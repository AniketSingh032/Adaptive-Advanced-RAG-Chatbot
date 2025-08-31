"""
Conversation Workflow Management Module.

This module orchestrates the conversation processing pipeline using LangGraph's
StateGraph implementation. It defines the workflow structure, node connections,
and routing logic for handling different types of queries.

Example:
    from src.utils.workflow import Workflow
    from langchain.schema import HumanMessage
    
    workflow = Workflow()
    graph = workflow.create_graph()
    config = {"configurable": {"thread_id": "abc123"}}
    
    for step in graph.stream(
        {"messages": [HumanMessage(content="What is RAG?")]},
        stream_mode="values",
        config=config
    ):
        step["messages"][-1].pretty_print()
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from src.utils.node import node
from src.utils.edge import edge
from src.utils.state import GraphState

#uncomment this if not running locally with Langsmith and langgraoh studio.
#memory = MemorySaver()

class Workflow:
    
    def create_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("router_node", node.router_node)
        workflow.add_node("general_answer_node", node.general_answer_node)
        workflow.add_node("relevant_docs_node", node.relevant_docs_node)
        workflow.add_node("answer_generation_node", node.answer_generation_node)

        # Set up workflow
        workflow.set_entry_point("router_node")

        # Add direct edges
        workflow.add_edge("relevant_docs_node", "answer_generation_node")
        workflow.add_edge("answer_generation_node", END)
        workflow.add_edge("general_answer_node", END)

        # Add conditional edges for routing
        workflow.add_conditional_edges(
            "router_node",
            edge.route_question,
            {
                "general_answer_node": "general_answer_node",
                "relevant_docs_node": "relevant_docs_node",
            }
        )
        # Uncomment this if not running locally with Langsmith and langgraoh studio since studio manages memory on its own.
        #return workflow.compile(checkpointer=memory)
        return workflow.compile()
    
graph = Workflow().create_graph()    