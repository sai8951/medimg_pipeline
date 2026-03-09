class MedImgPipelineError(Exception):
    """Base exception for medimg_pipeline."""


class UnsupportedInputError(MedImgPipelineError):
    """Raised when an unsupported input type is requested."""


class ConfigError(MedImgPipelineError):
    """Raised when config is invalid or inconsistent."""
