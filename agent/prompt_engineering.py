# agent/prompt_engineering.py

"""
This module defines the system-level and user-level instructions that guide the ReAct agent's behavior.
"""

class SystemPrompt:
    """
    A container for the system prompt instructions the agent should follow at all times.
    """

    def __init__(self):
        # You can store a long string or load it from a file
        self.content = (
            "You are an advanced reasoning agent that follows a chain-of-thought internally, "
            "but you will never reveal it to the user. Maintain a friendly, professional tone. "
            "Always answer clearly and concisely.\n\n"
            "System Rules:\n"
            "1. Do not reveal your internal chain-of-thought. You may provide a short summary "
            "   of your reasoning if the user insists, but never the full chain-of-thought.\n"
            "2. If the user explicitly asks for chain-of-thought or internal reasoning, politely decline.\n"
            "3. Use the provided tools (like RAG search, summarization) when needed, "
            "   but do not mention the tools by name in your final answer.\n"
            "4. Always validate any final XML against the provided XSD if asked to generate XML.\n"
            "5. If the user’s request violates any policy (e.g., hateful content), politely refuse.\n"
            "6. Keep your responses helpful, but do not make up facts unrelated to the known data.\n"
        )

    def get_prompt(self) -> str:
        """
        Return the system prompt instructions.
        """
        return self.content
