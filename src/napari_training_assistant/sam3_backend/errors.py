class SAM3BackendError(RuntimeError):
    """Base error for SAM3 backend failures."""


class SAM3ModelError(SAM3BackendError):
    """Raised when the SAM3 model cannot be loaded."""


class SAM3PromptError(SAM3BackendError):
    """Raised when prompt data is missing or invalid."""


class SAM3InferenceError(SAM3BackendError):
    """Raised when SAM3 inference fails."""