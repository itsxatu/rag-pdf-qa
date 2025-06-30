from langgraph.graph import StateGraph
from typing import TypedDict, Optional, List
from langchain.schema.document import Document

class GraphState(TypedDict):
    question: str
    answer: Optional[str]
    source_documents: Optional[List[Document]]

def build_langgraph_flow(qa_chain):
    def run_chain(state: GraphState) -> GraphState:
        question = state["question"]
        response = qa_chain.invoke({"question": question})

        return {
            "question": question,
            "answer": response["answer"],
            "source_documents": response.get("source_documents", [])
        }

    builder = StateGraph(GraphState)
    builder.add_node("qa_step", run_chain)
    builder.set_entry_point("qa_step")
    builder.set_finish_point("qa_step")
    return builder.compile()