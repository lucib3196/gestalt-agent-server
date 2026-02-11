import asyncio
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from langsmith import Client
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

from langgraph_server.gestalt_graphs.utils import extract_langsmith_prompt
from langgraph_server.gestalt_graphs.models import ExtractedQuestion
from langgraph_server.parsers import PDFMultiModalLLM
from langgraph_server.gestalt_graphs.utils import (
    save_graph_visualization,
    to_serializable,
)

client = Client()
prompt = extract_langsmith_prompt(client.pull_prompt("extract-all-questions"))

llm = init_chat_model(model="gpt-5-mini", model_provider="openai")


class State(BaseModel):
    lecture_pdf: str | Path
    questions: List[ExtractedQuestion] = []


class Response(BaseModel):
    questions: List[ExtractedQuestion]


async def extract_questions(state: State):
    processor = PDFMultiModalLLM()

    response = await processor.ainvoke(
        prompt=prompt,
        pdf_path=state.lecture_pdf,
        output_model=Response,
        llm=llm,
    )
    response = Response.model_validate(response)
    return {"questions": response.questions}


builder = StateGraph(State)
builder.add_node("extract_questions", extract_questions)

builder.add_edge(START, "extract_questions")

builder.add_edge("extract_questions", END)

graph = builder.compile()


if __name__ == "__main__":
    # Path to the lecture PDF
    pdf_path = Path(r"langgraph_server/gestalt_graphs/lecture_processing/ME135 Lecture Notes/11-17-25.pdf").resolve()

    output_path = Path(
        r"langgraph_server/gestalt_graphs/extract_question/output"
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
