# agent/tools.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.tools import BaseTool
from typing import List

class RAGSearchTool(BaseTool):
    """
    A tool that performs similarity search on a ChromaDB for relevant documents.
    """
    name = "search_tool"
    description = "Useful for searching scenario data using Chroma-based RAG."

    def __init__(self, collection_name: str = "scenario_collection", persist_directory: str = "chroma_db"):
        super().__init__()
        self.embedding_function = OpenAIEmbeddings()  # or local embedding model
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding_function
        )

    def _run(self, query: str) -> str:
        docs: List[Document] = self.vectorstore.similarity_search(query, k=3)
        combined = "\n\n".join([d.page_content for d in docs])
        return combined

    async def _arun(self, query: str) -> str:
        # Not implementing async version for this demo
        raise NotImplementedError("RAGSearchTool does not support async")

class SummarizeTool(BaseTool):
    """
    A tool that can be used to summarize text using the same LLM or a different chain.
    """
    name = "summarize_tool"
    description = "Useful for summarizing long text into a concise form."

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def _run(self, text: str) -> str:
        # In a real system, you'd create a SummarizationChain or a direct LLM call
        prompt = f"Please provide a concise summary of the following text:\n{text}"
        summary = self.llm.predict(prompt=prompt)
        return summary

    async def _arun(self, text: str) -> str:
        raise NotImplementedError("SummarizeTool does not support async")
