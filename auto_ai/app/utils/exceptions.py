class ModelSmithException(Exception):
    """Base exception class for all ModelSmith errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class AgentException(ModelSmithException):
    """Raised when a specific AI Agent fails during its execution step."""
    def __init__(self, agent_name: str, message: str, details: dict = None):
        extended_details = {"agent_name": agent_name, **(details or {})}
        super().__init__(f"Agent [{agent_name}] failed: {message}", extended_details)
        self.agent_name = agent_name

class DataValidationException(ModelSmithException):
    """Raised when dataset validation checks fail."""
    pass

class DataCleaningException(ModelSmithException):
    """Raised when data cleaning errors occur."""
    pass

class LLMException(ModelSmithException):
    """Raised when there is an issue calling the Gemini API."""
    pass
