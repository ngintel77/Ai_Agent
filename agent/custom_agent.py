# agent/custom_agent.py

import re
from langchain.agents import Tool, AgentExecutor, AgentOutputParser
from langchain.agents import LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish

# Import our new SystemPrompt
from agent.prompt_engineering import SystemPrompt

class ReActPromptTemplate(StringPromptTemplate):
    """
    A ReAct-like prompt template that merges system instructions with
    an example-based or instructions-based approach for deciding actions.
    """
    def __init__(self, input_variables, system_prompt: str):
        super().__init__(input_variables=input_variables)
        self.system_prompt = system_prompt

    def format(self, **kwargs) -> str:
        # The system prompt: overarching instructions
        system_instructions = self.system_prompt

        # Tools info is inserted from the agent call
        tools_info = kwargs.get("tools_info", "")
        user_input = kwargs.get("input", "")

        # Compose the final prompt
        # System instructions come first, followed by the ReAct methodology
        # The agent sees these instructions on every invocation
        prompt_text = f"""{system_instructions}

You have access to the following tools:
{tools_info}

Follow this format:

Question: the input question from the user
Thought: your internal reasoning (never to be revealed verbatim to the user)
Action: one and only one action to take, if needed
Action Input: the input to the action
Observation: the result of the action
Thought: more internal reasoning
Final Answer: the final answer to the user's question

Question: {user_input}"""

        return prompt_text

class SimpleOutputParser(AgentOutputParser):
    """
    Minimal parser to interpret the LLM's ReAct-style output into either
    an AgentAction or AgentFinish.
    """
    def parse(self, llm_output: str) -> AgentAction or AgentFinish:
        if "Final Answer:" in llm_output:
            result = llm_output.split("Final Answer:")[-1].strip()
            return AgentFinish(return_values={"output": result}, log=llm_output)

        # Otherwise, look for an Action
        action_match = re.search(r"Action:\s*(.*)", llm_output)
        if action_match:
            action = action_match.group(1).strip()
            action_input_match = re.search(r"Action Input:\s*(.*)", llm_output)
            if action_input_match:
                action_input = action_input_match.group(1).strip()
                return AgentAction(tool=action, tool_input=action_input, log=llm_output)

        # If no parseable action or final answer is found, default to finishing
        return AgentFinish(return_values={"output": llm_output.strip()}, log=llm_output)

def build_react_agent(llm, tools, memory) -> AgentExecutor:
    """
    Create an AgentExecutor that uses the ReAct pattern
    with a system prompt and a set of Tools.
    """
    # Load our system-level instructions
    system_prompt = SystemPrompt().get_prompt()

    # Create the custom ReAct prompt that merges system instructions
    prompt = ReActPromptTemplate(
        input_variables=["input", "tools_info"],
        system_prompt=system_prompt
    )
    output_parser = SimpleOutputParser()

    # Build the single-action agent from the LLM + custom prompt
    llm_chain = LLMSingleActionAgent(
        llm=llm,
        prompt=prompt,
        output_parser=output_parser,
        stop=["\nObservation:"]
    )

    # Convert our Tools to the format expected by LangChain
    tool_list = []
    tools_info = []
    for t in tools:
        tool_obj = Tool(
            name=t.name,
            func=t.run,
            description=t.description
        )
        tool_list.append(tool_obj)
        tools_info.append(f"{t.name}: {t.description}")

    tools_info_str = "\n".join(tools_info)

    # Assemble the AgentExecutor, passing short-term memory
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=llm_chain,
        tools=tool_list,
        verbose=True,
        memory=memory
    )

    # Return a callable function
    def custom_call(input_str: str) -> str:
        return agent_executor.run(
            input=input_str,
            tools_info=tools_info_str
        )

    return custom_call
