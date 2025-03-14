# ingest_scenarios.py
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

def ingest_scenarios(scenario_dir="data/scenario_files", persist_directory="chroma_db"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    embedding_function = OpenAIEmbeddings()

    docs_to_store = []
    for filename in os.listdir(scenario_dir):
        if filename.endswith(".txt") or filename.endswith(".md") or filename.endswith(".docx"):
            with open(os.path.join(scenario_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                docs_to_store.append(chunk)

    store = Chroma.from_texts(
        docs_to_store,
        embedding_function,
        persist_directory=persist_directory,
        collection_name="scenario_collection",
    )
    store.persist()

if __name__ == "__main__":
    ingest_scenarios()
