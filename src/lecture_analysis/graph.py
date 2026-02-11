# --- Standard Library ---
import asyncio
from pathlib import Path

# --- Third-Party ---
from pydantic import BaseModel
from langsmith import Client
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# --- Local Application ---
from langgraph_server.gestalt_graphs.models import LectureAnalysis
from langgraph_server.parsers import PDFMultiModalLLM
from langgraph_server.gestalt_graphs.utils import (
    save_graph_visualization,
    to_serializable,
    extract_langsmith_prompt,
)

client = Client()

llm = init_chat_model(model="gpt-5-mini", model_provider="openai")
prompt = extract_langsmith_prompt(client.pull_prompt("lecture_analysis"))


class State(BaseModel):
    lecture_pdf: str | Path
    analysis: LectureAnalysis | None = None


async def initial_analysis(state: State):
    processor = PDFMultiModalLLM()

    response = await processor.ainvoke(
        prompt=prompt,
        pdf_path=state.lecture_pdf,
        output_model=LectureAnalysis,
        llm=llm,
    )
    response = LectureAnalysis.model_validate(response)
    return {"analysis": response}


builder = StateGraph(State)
builder.add_node("extract_derivations", initial_analysis)
builder.add_edge(START, "extract_derivations")
builder.add_edge("extract_derivations", END)
graph = builder.compile()


if __name__ == "__main__":
    # Path to the lecture PDF
    pdf_path = Path(r"langgraph_server/data/Lecture_02_03.pdf").resolve()

    output_path = Path(
        r"langgraph_server/gestalt_graphs/lecture_analysis/output"
    ).resolve()

    save_path = output_path
    save_graph_visualization(graph, save_path, "graph.png")

    # Create graph input state
    graph_input = State(lecture_pdf=pdf_path)

    # Run the async graph and print the response
    try:
        response = asyncio.run(graph.ainvoke(graph_input))
        print("\n--- Graph Response ---")
        print(response)
        import json

        data_path = save_path / "output.json"
        data_path.write_text(json.dumps(to_serializable(response)))
    except Exception as e:
        print("\n❌ Error while running graph:")
        print(e)
