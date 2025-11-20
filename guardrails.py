class HealthGuardrail:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def check_input_safety(self, query: str) -> bool:
        """
        Returns True if the query is safe (health-related).
        Returns False if the query is unsafe (politics, coding, etc.).
        """
        system_prompt = """
        You are a content moderator for a medical bot. 
        Classify the following user query.
        If it is related to health, medicine, biology, or the human body, respond with 'SAFE'.
        If it is unrelated (e.g., coding, politics, general chit-chat) or malicious, respond with 'UNSAFE'.
        Respond ONLY with the word 'SAFE' or 'UNSAFE'.
        """
        try:
            response = self.llm.invoke(f"{system_prompt}\nUser Query: {query}")
            return "SAFE" in response.content.strip().upper()
        except Exception:
            return True # Fail open if API has issues

    def check_output_safety(self, response_text: str) -> bool:
        """Returns True if the output contains no dangerous advice."""
        system_prompt = """
        You are a medical safety auditor. Check the following response.
        Does it contain harmful, illegal, or clearly dangerous medical advice?
        Respond 'SAFE' if it looks like standard information.
        Respond 'UNSAFE' if it promotes self-harm or extremely dangerous practices.
        """
        try:
            audit = self.llm.invoke(f"{system_prompt}\nResponse: {response_text}")
            return "SAFE" in audit.content.strip().upper()
        except Exception:
            return True