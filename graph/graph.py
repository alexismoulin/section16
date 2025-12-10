from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEBSEARCH
from graph.nodes import retrieve, generate, grade_documents, web_search
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    if hallucination_score.binary_score:
        print("---GENERATION GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_score.binary_score:
            print("---DECISION: GENERATION ANSWER QUESTION---")
            return "useful"
        else:
            print("---GENERATION DOES NOT ADDRESS THE QUESTION")
            return "not useful"
        
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"


def route_query(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEBSEARCH---")
        return WEBSEARCH
    else :
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


workflow = StateGraph(state_schema=GraphState) # type: ignore[arg-type]

workflow.add_node(RETRIEVE, retrieve) # type: ignore[arg-type]
workflow.add_node(GRADE_DOCUMENTS, grade_documents) # type: ignore[arg-type]
workflow.add_node(GENERATE, generate) # type: ignore[arg-type]
workflow.add_node(WEBSEARCH, web_search) # type: ignore[arg-type]

workflow.set_conditional_entry_point(path=route_query, path_map={WEBSEARCH:WEBSEARCH, RETRIEVE: RETRIEVE})

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    source=GRADE_DOCUMENTS,
    path=decide_to_generate,
    path_map={WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)
workflow.add_conditional_edges(
    source=GENERATE,
    path=grade_generation_grounded_in_documents_and_question,
    path_map={"not supported": GENERATE, "useful": END, "not useful": WEBSEARCH}
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
