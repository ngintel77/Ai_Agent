# server.py

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from agent.llm_manager import LLMManager
from agent.memory_manager import MemoryManager
from agent.tools import RAGSearchTool, SummarizeTool
from agent.custom_agent import build_react_agent
from agent.review_manager import ReviewManager
from agent.xml_generator import XMLGenerator

app = FastAPI()

# Initialize global objects (in real-world, handle them more elegantly)
llm_manager = LLMManager(
    use_azure_openai=True,
    azure_openai_deployment_name="your_deployment_name",
    azure_openai_api_key="your_azure_api_key"
)
llm = llm_manager.get_llm()

memory_manager = MemoryManager(
    pg_host="localhost",
    pg_port=5432,
    pg_database="mydb",
    pg_user="myuser",
    pg_password="mypassword"
)
short_term_memory = memory_manager.get_short_term_memory()

search_tool = RAGSearchTool()
summarize_tool = SummarizeTool(llm)
tools = [search_tool, summarize_tool]

react_agent = build_react_agent(llm, tools, short_term_memory)
review_manager = ReviewManager(llm)
xml_gen = XMLGenerator(xsd_path="agent/schemas/scenario.xsd")

class UserQuery(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(payload: UserQuery):
    user_query = payload.query
    # ReAct agent call
    intermediate_response = react_agent(user_query)

    # Build final XML
    try:
        xml_output = xml_gen.build_xml(
            scenario_data=intermediate_response,
            user_requirements=user_query,
            additional_metadata="ReAct-based approach used."
        )
        # Store in DB
        memory_manager.add_long_term_memory(key=user_query, value=xml_output)

        # Self-Review
        review_text = review_manager.generate_review(user_query, xml_output)
        memory_manager.add_long_term_memory(key=f"{user_query}_review", value=review_text)

        return {"xml": xml_output, "review": review_text}
    except ValueError as e:
        return {"error": str(e)}

class FeedbackPayload(BaseModel):
    user_query: str
    feedback_text: str

@app.post("/feedback")
def feedback_endpoint(payload: FeedbackPayload):
    memory_manager.store_feedback(payload.user_query, payload.feedback_text)
    return {"status": "Feedback recorded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
