import os
from langchain_astradb import AstraDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.settings import get_settings
from src.document_loaders.firebase_loader import FirebaseLectureDocumentLoader


settings = get_settings()
embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
vector_store = AstraDBVectorStore(
    collection_name="me118_lecture_fb",
    embedding=embeddings,
    api_endpoint=settings.ASTRA_DB_API_ENDPOINT,
    token=settings.ASTRA_DB_APPLICATION_TOKEN,
)

if __name__ == "__main__":
    docs = FirebaseLectureDocumentLoader(
        prefix="me118_winter_2026/lectures"
    ).load_and_split()
    new_docs = []
    for d in docs:
        doc_id = d.id
        if not doc_id:
            raise ValueError(f"Document does not contain an id {d}")
        existing_document = vector_store.get_by_document_id(doc_id)
        if existing_document:
            print("Existing meta", existing_document.metadata, "\n")
            print("New meta",existing_document.metadata,"\n")
            print(d.metadata)
            print("Skipping doc already exist")
            continue
        new_docs.append(d)
    if new_docs:
        print(f"Adding docs lenght:  {len(new_docs)}")
        vector_store.add_documents(new_docs)
