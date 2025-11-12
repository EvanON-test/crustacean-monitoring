"""
Core pipeline orchestration modules.

This package contains the base pipeline class and implementations for
offline (batch video processing) and real-time (live camera) modes.
"""

from crustacean.core.pipeline import Pipeline
from crustacean.core.offline_pipeline import OfflinePipeline
from crustacean.core.realtime_pipeline import RealtimePipeline

__all__ = [
    "Pipeline",
    "OfflinePipeline",
    "RealtimePipeline",
]
