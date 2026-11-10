class ProviderBaseError(Exception):
    pass

class FourGetError(ProviderBaseError):

    TRANSIENT = "transient"
    BLOCKED = "blocked"
    DOWN = "down"
    INVALID = "invalid"
    NPT_ERROR_TERMS = (
        "malformed",
        "nextpagetoken",
        "search term is empty"
    )

    @classmethod
    def kind_from_message(cls, message: str) -> str:
        msg = message.lower()
        if any(hint in msg for hint in cls.NPT_ERROR_TERMS):
            return FourGetError.INVALID
        return FourGetError.BLOCKED

    def __init__(self, message: str, kind : str):
        super().__init__(message)
        self.kind = kind
