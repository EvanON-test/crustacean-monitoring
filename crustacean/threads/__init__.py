"""
Thread management for real-time pipeline.

This package contains thread implementations for parallel processing
in the real-time pipeline mode.
"""

from crustacean.threads.analysis_thread import AnalysisThread
from crustacean.threads.detection_thread import DetectionThread
from crustacean.threads.save_thread import save_detection

__all__ = [
    "AnalysisThread",
    "DetectionThread",
    "save_detection",
]
