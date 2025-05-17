
class BaseAppException(Exception):
    def __init__(self, message: str | None, *, details: str | None = None, exc_info: Exception | None = None):
        self.message = message
        self.details = details
        self.exc_info = exc_info
        full_msg = f"{message}"
        if details:
            full_msg += f" | Details: {details}"
        if exc_info:
            full_msg += f" | Caused by: {repr(exc_info)}"
        super().__init__(full_msg)

class ArgumentException(BaseAppException):
    pass
