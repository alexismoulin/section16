from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("----WEBSEARCH----")
    question = state["question"]
    documents = state.get("documents") or []

    tavily_response = web_search_tool.invoke({"query": question})

    if isinstance(tavily_response, dict) and tavily_response.get("error"):
        # Surface web-search failures instead of hard-crashing the graph.
        error_message = f"Tavily search failed: {tavily_response['error']}"
        print(error_message)
        web_results = Document(page_content=error_message)
        documents.append(web_results)
        return {"question": question, "documents": documents}

    results: List[str] = []
    if isinstance(tavily_response, dict):
        for tavily_result in tavily_response.get("results", []):
            if isinstance(tavily_result, dict) and tavily_result.get("content"):
                results.append(tavily_result["content"])
    elif isinstance(tavily_response, list):
        for tavily_result in tavily_response:
            if isinstance(tavily_result, dict) and tavily_result.get("content"):
                results.append(tavily_result["content"])

    joined_tavily_results = "\n".join(results)
    web_results = Document(page_content=joined_tavily_results)
    documents.append(web_results)

    return {"question": question, "documents": documents}
