# agent/review_manager.py

class ReviewManager:
    def __init__(self, llm):
        self.llm = llm

    def generate_review(self, user_query: str, xml_output: str) -> str:
        """
        The agent critiques its own output, suggesting potential improvements for next time.
        """
        prompt = f"""
        The user asked: {user_query}
        The agent produced this XML: {xml_output}

        Please critique this output, noting any possible improvements, missing details,
        or alternative approaches that might yield a better result in future interactions.
        """
        review = self.llm.predict(prompt=prompt)
        return review
