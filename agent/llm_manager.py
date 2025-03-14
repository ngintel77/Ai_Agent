# agent/llm_manager.py
import os
from typing import Optional

class LLMManager:
    def __init__(
        self,
        use_azure_openai: bool = True,
        azure_openai_deployment_name: Optional[str] = None,
        azure_openai_api_key: Optional[str] = None,
        ollama_endpoint: Optional[str] = None
    ):
        """
        Configure LLM. If `use_azure_openai` is True, we use Azure,
        otherwise we assume Ollama. Adjust as needed for your environment.
        """
        self.use_azure_openai = use_azure_openai
        self.azure_openai_deployment_name = azure_openai_deployment_name
        self.azure_openai_api_key = azure_openai_api_key
        self.ollama_endpoint = ollama_endpoint
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        if self.use_azure_openai:
            from langchain.chat_models import AzureChatOpenAI
            self.llm = AzureChatOpenAI(
                deployment_name=self.azure_openai_deployment_name,
                openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                openai_api_version="2023-03-15-preview",
                openai_api_key=self.azure_openai_api_key,
                temperature=0.2
            )
        else:
            # Adjust for Ollama integration or other providers
            from langchain.llms import Ollama
            self.llm = Ollama(
                base_url=self.ollama_endpoint,
                model="llama2"  # Example
            )

    def get_llm(self):
        return self.llm
