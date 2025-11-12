"""
Machine learning model interfaces.

This package contains the base model class and implementations for all
four stages of the crustacean detection pipeline:
- Binary Classifier: Detects crustacean presence
- Frame Selector: Selects highest quality frames
- Object Detector: Localizes crustaceans and extracts ROI
- Keypoint Detector: Identifies anatomical landmarks
"""

from crustacean.models.base_model import BaseModel
from crustacean.models.binary_classifier import BinaryClassifier
from crustacean.models.frame_selector import FrameSelector
from crustacean.models.object_detector import ObjectDetector
from crustacean.models.keypoint_detector import KeypointDetector

__all__ = [
    "BaseModel",
    "BinaryClassifier",
    "FrameSelector",
    "ObjectDetector",
    "KeypointDetector",
]
