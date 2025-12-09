"""
Thread management for real-time pipeline.

This package contains thread implementations for parallel processing
in the real-time pipeline mode.

Classes:
    AnalysisThread: Processes frames through Binary Classifier and Frame Selector
    DetectionThread: Processes frames through Object Detector
    DetectionResult: Container for detection results

Functions:
    save_detection: Saves detection results to disk
"""

from crustacean.threads.analysis_thread import AnalysisThread
from crustacean.threads.detection_thread import DetectionThread, DetectionResult
from crustacean.threads.save_thread import save_detection

__all__ = [
    "AnalysisThread",
    "DetectionThread",
    "DetectionResult",
    "save_detection",
]
