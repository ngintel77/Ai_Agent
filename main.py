# main.py

from agent.llm_manager import LLMManager
from agent.memory_manager import MemoryManager
from agent.tools import RAGSearchTool, SummarizeTool
from agent.custom_agent import build_react_agent
from agent.review_manager import ReviewManager
from agent.xml_generator import XMLGenerator

def main():
    # 1. LLM Setup
    llm_manager = LLMManager(
        use_azure_openai=True,
        azure_openai_deployment_name="your_azure_deployment_name",
        azure_openai_api_key="your_azure_api_key"
    )
    llm = llm_manager.get_llm()

    # 2. Memory (Postgres for long-term + conversation buffer)
    memory_manager = MemoryManager(
        pg_host="localhost",
        pg_port=5432,
        pg_database="mydb",
        pg_user="myuser",
        pg_password="mypassword"
    )
    short_term_memory = memory_manager.get_short_term_memory()

    # 3. Tools for ReAct
    search_tool = RAGSearchTool(collection_name="scenario_collection", persist_directory="chroma_db")
    summarize_tool = SummarizeTool(llm)
    tools = [search_tool, summarize_tool]

    # 4. Build ReAct agent with our prompt-engineered system instructions
    react_agent = build_react_agent(llm, tools, short_term_memory)

    # 5. Review Manager for self-critique
    review_manager = ReviewManager(llm)

    # 6. XML Generator with XSD validation
    xml_generator = XMLGenerator(xsd_path="agent/schemas/scenario.xsd")

    # 7. Example flow
    user_query = input("Enter your request: ")
    intermediate_response = react_agent(user_query)

    print("\n[DEBUG] ReAct agent intermediate response:")
    print(intermediate_response)

    try:
        # Generate final XML from the intermediate data & user query
        scenario_data = intermediate_response
        xml_output = xml_generator.build_xml(
            scenario_data=scenario_data,
            user_requirements=user_query,
            additional_metadata="ReAct with system prompt"
        )

        print("\n=== Generated Validated XML ===")
        print(xml_output)

        # Store final XML in long-term memory
        memory_manager.add_long_term_memory(key=user_query, value=xml_output)

        # Self-review step
        review_text = review_manager.generate_review(user_query, xml_output)
        print("\n=== Agent's Self-Review ===")
        print(review_text)

        # Optionally store the self-review
        memory_manager.add_long_term_memory(key=f"{user_query}_review", value=review_text)

        # Collect user feedback
        feedback = input("\nPlease provide any feedback (or press Enter to skip): ")
        if feedback.strip():
            memory_manager.store_feedback(user_query, feedback)
            print("Feedback recorded. Thank you!")

    except ValueError as e:
        print(f"Error generating or validating XML: {e}")

if __name__ == "__main__":
    main()
