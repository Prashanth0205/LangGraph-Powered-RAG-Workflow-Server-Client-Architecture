from typing import TypedDict, List

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes: 
        question: question
        generation: LLM generation
        documents: List of documents
    """
    input: str
    generation: str
    documents: str