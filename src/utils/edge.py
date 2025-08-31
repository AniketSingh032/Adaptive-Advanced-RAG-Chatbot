class Edge:
    """
    Edge Router for Question Processing Pipeline.
    
    This class handles the routing logic in the RAG (Retrieval-Augmented Generation) system,
    determining whether questions should be:
    - Sent directly to general answer generation ("general_answer_node")
    - Processed through document retrieval first ("relevant_docs_node")
    
    The routing decision is based on the question's category stored in the state dictionary.
    
    Attributes:
        None
        
    Example:
        edge = Edge()
        next_node = edge.route_question({"category": "general"})
        # Returns: "general_answer_node"
    """

    def route_question(self, state):
        """Route the question to the appropriate node based on its category."""
        if state["category"] == "general":
            return "general_answer_node"
        else:
            return "relevant_docs_node"

edge = Edge()