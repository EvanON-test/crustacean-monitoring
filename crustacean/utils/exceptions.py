"""
Custom exceptions - placeholder for Task 4.
"""

class CrustaceanError(Exception):
    """Base exception."""
    pass

class ConfigurationError(CrustaceanError):
    """Configuration error."""
    pass

class ModelLoadError(CrustaceanError):
    """Model loading error."""
    pass

class ModelNotLoadedError(CrustaceanError):
    """Model not loaded error."""
    pass

class CameraInitError(CrustaceanError):
    """Camera initialization error."""
    pass

class InferenceError(CrustaceanError):
    """Inference error."""
    pass

class ThreadError(CrustaceanError):
    """Thread error."""
    pass
