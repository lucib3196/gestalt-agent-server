from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
embeddings = OpenAIEmbeddings()
vector_store = AstraDBVectorStore(
    collection_name="me135_lecture",
    embedding=embeddings,
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
)

if __name__ == "__main__":

    from dotenv import load_dotenv
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from langchain_openai import OpenAIEmbeddings

    from ME135Agent.document_loader import LectureDocumentLoader

    loader = LectureDocumentLoader(
        root=r"assets/ME135Lecture",
        metadata={"course": "ME135 Transport Phenomena", "professor": "Sundar"},
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.add_documents(documents=all_splits)
