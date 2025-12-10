from typing import List, TypedDict, Optional

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of one graph
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: Optional[List[Document]]
