"""
Save Detection functionality for real-time pipeline.

This module provides the save_detection function which handles saving
detection results including frames and keypoint data to disk.
"""

import os
import csv
import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from crustacean.utils.logging_setup import get_logger
from crustacean.utils.exceptions import InferenceError


logger = get_logger(__name__)


def save_detection(
    frame: np.ndarray,
    roi: np.ndarray,
    confidence: float,
    frame_number: int,
    config,
    kd_model=None
) -> Optional[str]:
    """
    Save a detection result to disk.
    
    This function saves the original frame as a JPG image, processes
    the ROI through the keypoint detector, and saves the keypoint
    coordinates to a CSV file.
    
    Args:
        frame: Original BGR frame
        roi: Cropped region of interest for keypoint detection
        confidence: Detection confidence score
        frame_number: Frame number for tracking
        config: Configuration object with output paths
        kd_model: Optional pre-loaded KeypointDetector model.
                  If None, will load/unload model internally.
    
    Returns:
        Path to the detection directory, or None on failure
        
    Example:
        >>> result_path = save_detection(
        ...     frame, roi, 0.85, 1234, config
        ... )
        >>> print(f"Saved to: {result_path}")
    """
    try:
        # Get output directory from config
        output_dir = Path(config.get('output.detections_dir', './realtime_frames'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique naming
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create detection directory
        detection_dir = output_dir / f"{timestamp}_Detection"
        detection_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving detection to {detection_dir}")
        
        # Save original frame
        frame_path = _save_frame(detection_dir, timestamp, frame)
        
        if frame_path is None:
            logger.error("Failed to save frame")
            return None
        
        # Process ROI through keypoint detector and save
        keypoints_path = _process_and_save_keypoints(
            detection_dir, timestamp, roi, config, kd_model
        )
        
        if keypoints_path is None:
            logger.warning("Failed to save keypoints")
            # Still return success since frame was saved
        
        # Save detection metadata
        _save_metadata(detection_dir, timestamp, confidence, frame_number)
        
        logger.info(
            f"Detection saved: frame={frame_number}, confidence={confidence:.3f}"
        )
        
        return str(detection_dir)
        
    except Exception as e:
        logger.error(f"Failed to save detection: {e}")
        return None


def _save_frame(
    detection_dir: Path,
    timestamp: str,
    frame: np.ndarray
) -> Optional[str]:
    """
    Save the original frame as a JPG image.
    
    Args:
        detection_dir: Directory to save to
        timestamp: Timestamp string for filename
        frame: BGR frame to save
        
    Returns:
        Path to saved image, or None on failure
    """
    try:
        filename = f"{timestamp}_screenshot.jpg"
        filepath = detection_dir / filename
        
        success = cv2.imwrite(str(filepath), frame)
        
        if success:
            logger.debug(f"Saved frame to {filepath}")
            return str(filepath)
        else:
            logger.error(f"cv2.imwrite failed for {filepath}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to save frame: {e}")
        return None


def _process_and_save_keypoints(
    detection_dir: Path,
    timestamp: str,
    roi: np.ndarray,
    config,
    kd_model=None
) -> Optional[str]:
    """
    Process ROI through keypoint detector and save results.
    
    Args:
        detection_dir: Directory to save to
        timestamp: Timestamp string for filename
        roi: Region of interest for keypoint detection
        config: Configuration object
        kd_model: Optional pre-loaded KeypointDetector
        
    Returns:
        Path to saved CSV, or None on failure
    """
    try:
        # Load keypoint detector if not provided
        model_loaded = False
        
        if kd_model is None:
            from crustacean.models.keypoint_detector import KeypointDetector
            kd_model = KeypointDetector(config)
            kd_model.load()
            model_loaded = True
        
        try:
            # Process ROI - reshape for batch processing
            roi_batch = np.expand_dims(roi, axis=0) if roi.ndim == 3 else roi
            
            # Run keypoint detection
            keypoints = kd_model.predict(roi_batch)
            
            # Save keypoints to CSV
            csv_path = _save_keypoints_csv(detection_dir, timestamp, keypoints)
            
            return csv_path
            
        finally:
            # Unload model if we loaded it
            if model_loaded:
                kd_model.unload()
                
    except Exception as e:
        logger.error(f"Failed to process keypoints: {e}")
        return None


def _save_keypoints_csv(
    detection_dir: Path,
    timestamp: str,
    keypoints: np.ndarray
) -> Optional[str]:
    """
    Save keypoint coordinates to CSV file.
    
    Args:
        detection_dir: Directory to save to
        timestamp: Timestamp string for filename
        keypoints: Keypoint coordinates array
        
    Returns:
        Path to saved CSV, or None on failure
    """
    try:
        filename = f"{timestamp}_keypoints.csv"
        filepath = detection_dir / filename
        
        # Flatten keypoints for CSV writing
        # Keypoint detector returns 7 keypoints with 2 coords each
        flattened = keypoints.flatten()
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header with keypoint names
            headers = [
                'crab_left_x1', 'crab_left_y1',
                'crab_right_x2', 'crab_right_y2',
                'left_eye_x3', 'left_eye_y3',
                'right_eye_x4', 'right_eye_y4',
                'carapace_end_x5', 'carapace_end_y5',
                'tail_end_x6', 'tail_end_y6',
                'last_segment_x7', 'last_segment_y7'
            ]
            
            writer.writerow(headers)
            writer.writerow(flattened)
        
        logger.debug(f"Saved keypoints to {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Failed to save keypoints CSV: {e}")
        return None


def _save_metadata(
    detection_dir: Path,
    timestamp: str,
    confidence: float,
    frame_number: int
) -> Optional[str]:
    """
    Save detection metadata to a text file.
    
    Args:
        detection_dir: Directory to save to
        timestamp: Timestamp string for filename
        confidence: Detection confidence score
        frame_number: Frame number
        
    Returns:
        Path to saved metadata file, or None on failure
    """
    try:
        filename = f"{timestamp}_metadata.txt"
        filepath = detection_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"frame_number: {frame_number}\n")
            f.write(f"confidence: {confidence:.4f}\n")
        
        logger.debug(f"Saved metadata to {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        return None
