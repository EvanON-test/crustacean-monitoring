"""
Detection Thread for real-time pipeline processing.

This module implements the DetectionThread which processes frames through
the Object Detector model in a background thread.
"""

import queue
from threading import Thread
from typing import Tuple, Optional

import numpy as np

from crustacean.utils.logging_setup import get_logger
from crustacean.utils.exceptions import ThreadError


class DetectionThread(Thread):
    """
    Background thread for Object Detection.
    
    This thread receives frames from the analysis thread and processes
    them through the Object Detector to locate crustaceans and extract
    regions of interest (ROIs).
    
    Attributes:
        frame_queue: Input queue for frames to process
        result_queue: Output queue for detection results
        od_model: Object Detector model instance
        running: Flag to control thread execution
        
    Example:
        >>> frame_queue = Queue(maxsize=1)
        >>> result_queue = Queue(maxsize=1)
        >>> thread = DetectionThread(frame_queue, result_queue, od_model)
        >>> thread.start()
        >>> frame_queue.put((frame, frame_number))
        >>> result = result_queue.get()
        >>> thread.stop()
        >>> thread.join()
    """
    
    def __init__(
        self,
        frame_queue: queue.Queue,
        result_queue: queue.Queue,
        od_model,
        config=None
    ):
        """
        Initialize the Detection Thread.
        
        Args:
            frame_queue: Queue to receive frames from analysis thread
            result_queue: Queue to send detection results to main thread
            od_model: Object Detector model instance (should be loaded)
            config: Optional configuration object
        """
        super().__init__(name="DetectionThread", daemon=True)
        
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.od = od_model
        self.config = config
        self.running = True
        self.logger = get_logger(self.__class__.__name__)
        
        # Get confidence threshold from config
        self.confidence_threshold = 0.75
        if config:
            self.confidence_threshold = config.get(
                'models.object_detector.confidence_threshold', 0.75
            )
        
        self.logger.info("DetectionThread initialized")
    
    def run(self) -> None:
        """
        Main thread execution loop.
        
        Continuously processes frames from the frame queue
        until stop() is called.
        """
        self.logger.info("DetectionThread started")
        
        try:
            while self.running:
                try:
                    # Get frame with timeout for periodic stop checks
                    frame_data = self.frame_queue.get(timeout=2)
                    
                    if frame_data is None:
                        continue
                    
                    frame, frame_number = frame_data
                    self.logger.debug(f"Processing frame {frame_number}")
                    
                    self._process_frame(frame, frame_number)
                    
                except queue.Empty:
                    # Timeout - check if we should stop
                    continue
                    
        except Exception as e:
            self.logger.exception(f"DetectionThread failed: {e}")
            
        finally:
            self.logger.info("DetectionThread stopped")
    
    def _process_frame(
        self, 
        frame: np.ndarray, 
        frame_number: int
    ) -> None:
        """
        Process a single frame through Object Detection.
        
        Runs the frame through the Object Detector and sends results
        to the result queue if confidence is above threshold.
        
        Args:
            frame: BGR frame to process
            frame_number: Frame number for tracking
        """
        try:
            # Run object detection
            roi, confidence, class_idx = self.od.predict(frame)
            
            self.logger.debug(
                f"Frame {frame_number}: confidence={confidence:.3f}, class={class_idx}"
            )
            
            # Send result to main thread
            result = DetectionResult(
                frame=frame,
                roi=roi,
                confidence=confidence,
                class_index=class_idx,
                frame_number=frame_number
            )
            
            try:
                self.result_queue.put_nowait(result)
                self.logger.debug(f"Detection result queued for frame {frame_number}")
            except queue.Full:
                self.logger.warning("Result queue full - dropping detection")
                
        except Exception as e:
            self.logger.error(f"Object detection failed for frame {frame_number}: {e}")
    
    def stop(self) -> None:
        """
        Signal the thread to stop.
        
        The thread will finish processing the current frame before stopping.
        """
        self.logger.info("Stopping DetectionThread")
        self.running = False


class DetectionResult:
    """
    Container for object detection results.
    
    Attributes:
        frame: Original BGR frame
        roi: Cropped region of interest
        confidence: Detection confidence score
        class_index: Detected class index (0=crab, 1=lobster)
        frame_number: Frame number for tracking
    """
    
    def __init__(
        self,
        frame: np.ndarray,
        roi: np.ndarray,
        confidence: float,
        class_index: int,
        frame_number: int
    ):
        """
        Initialize detection result.
        
        Args:
            frame: Original BGR frame
            roi: Cropped region of interest
            confidence: Detection confidence score
            class_index: Detected class index
            frame_number: Frame number for tracking
        """
        self.frame = frame
        self.roi = roi
        self.confidence = confidence
        self.class_index = class_index
        self.frame_number = frame_number
    
    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        """
        Check if detection meets confidence threshold.
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            True if confidence >= threshold
        """
        return self.confidence >= threshold
    
    def __repr__(self) -> str:
        return (
            f"DetectionResult(frame={self.frame_number}, "
            f"confidence={self.confidence:.3f}, class={self.class_index})"
        )
