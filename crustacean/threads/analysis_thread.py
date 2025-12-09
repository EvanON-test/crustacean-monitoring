"""
Analysis Thread for real-time pipeline processing.

This module implements the AnalysisThread which processes frames through
the Binary Classifier and Frame Selector models in a background thread.
"""

import os
import queue
import tempfile
from threading import Thread
from typing import List, Tuple, Optional

import cv2
import numpy as np

from crustacean.utils.logging_setup import get_logger
from crustacean.utils.exceptions import ThreadError


class AnalysisThread(Thread):
    """
    Background thread for Binary Classification and Frame Selection.
    
    This thread receives batches of frames from the main pipeline,
    processes them through the Binary Classifier to detect crustacean
    presence, then uses the Frame Selector to identify the best quality
    frames for object detection.
    
    Attributes:
        analysis_queue: Input queue for frame batches
        detection_queue: Output queue for selected frames
        bc_model: Binary Classifier model instance
        fs_model: Frame Selector model instance
        running: Flag to control thread execution
        
    Example:
        >>> analysis_queue = Queue(maxsize=1)
        >>> detection_queue = Queue(maxsize=1)
        >>> thread = AnalysisThread(
        ...     analysis_queue, detection_queue, bc_model, fs_model
        ... )
        >>> thread.start()
        >>> analysis_queue.put((frames, start_frame))
        >>> # ... later
        >>> thread.stop()
        >>> thread.join()
    """
    
    def __init__(
        self,
        analysis_queue: queue.Queue,
        detection_queue: queue.Queue,
        bc_model,
        fs_model,
        config=None
    ):
        """
        Initialize the Analysis Thread.
        
        Args:
            analysis_queue: Queue to receive frame batches from main thread
            detection_queue: Queue to send selected frames to detection thread
            bc_model: Binary Classifier model instance (should be loaded)
            fs_model: Frame Selector model instance (should be loaded)
            config: Optional configuration object
        """
        super().__init__(name="AnalysisThread", daemon=True)
        
        self.analysis_queue = analysis_queue
        self.detection_queue = detection_queue
        self.bc = bc_model
        self.fs = fs_model
        self.config = config
        self.running = True
        self.logger = get_logger(self.__class__.__name__)
        
        self.logger.info("AnalysisThread initialized")
    
    def run(self) -> None:
        """
        Main thread execution loop.
        
        Continuously processes frame batches from the analysis queue
        until stop() is called.
        """
        self.logger.info("AnalysisThread started")
        
        try:
            while self.running:
                try:
                    # Get frame batch with timeout for periodic stop checks
                    frame_data = self.analysis_queue.get(timeout=2)
                    
                    if frame_data is None:
                        continue
                    
                    frames, start_frame = frame_data
                    self.logger.debug(
                        f"Processing {len(frames)} frames starting at {start_frame}"
                    )
                    
                    self._process_frames(frames, start_frame)
                    
                except queue.Empty:
                    # Timeout - check if we should stop
                    continue
                    
        except Exception as e:
            self.logger.exception(f"AnalysisThread failed: {e}")
            
        finally:
            self.logger.info("AnalysisThread stopped")
    
    def _process_frames(
        self, 
        frames: List[np.ndarray], 
        start_frame: int
    ) -> None:
        """
        Process a batch of frames through BC and FS.
        
        Creates a temporary video from the frames, runs Binary Classification
        to detect crustacean presence, then runs Frame Selection to find
        the best quality frame for object detection.
        
        Args:
            frames: List of BGR frames to process
            start_frame: Frame number of the first frame in the batch
        """
        temp_video_path = None
        
        try:
            # Create temporary video from frames
            temp_video_path = self._create_temp_video(frames)
            
            if temp_video_path is None:
                self.logger.warning("Failed to create temporary video")
                return
            
            # Run Binary Classification
            signal = self._run_binary_classifier(temp_video_path)
            
            if signal is None:
                return
            
            # Check for positive detections
            positive_frames = sum(signal)
            if positive_frames == 0:
                self.logger.debug("No crustacean detected - skipping FS")
                return
            
            self.logger.debug(f"BC detected presence in {positive_frames} frames")
            
            # Run Frame Selection
            selected_indices = self._run_frame_selector(signal, temp_video_path)
            
            if selected_indices is None:
                return
            
            # Select best frame and send to detection queue
            self._send_best_frame(frames, selected_indices, start_frame)
            
        except Exception as e:
            self.logger.error(f"Error processing frames: {e}")
            
        finally:
            # Clean up temporary video
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp video: {e}")
    
    def _create_temp_video(self, frames: List[np.ndarray]) -> Optional[str]:
        """
        Create a temporary video file from frames.
        
        Args:
            frames: List of BGR frames
            
        Returns:
            Path to temporary video file, or None on failure
        """
        try:
            temp_video = tempfile.mktemp(suffix=".mp4")
            height, width = frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_video, fourcc, 15.0, (width, height))
            
            for frame in frames:
                writer.write(frame)
            
            writer.release()
            
            self.logger.debug(f"Created temp video: {temp_video}")
            return temp_video
            
        except Exception as e:
            self.logger.error(f"Failed to create temp video: {e}")
            return None
    
    def _run_binary_classifier(self, video_path: str) -> Optional[np.ndarray]:
        """
        Run Binary Classifier on video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Binary signal array, or None on failure
        """
        try:
            capture = cv2.VideoCapture(video_path)
            
            if not capture.isOpened():
                self.logger.error("Failed to open video for BC")
                return None
            
            try:
                signal = self.bc.predict(capture)
                self.logger.debug(f"BC signal: {signal}")
                return signal
            finally:
                capture.release()
                
        except Exception as e:
            self.logger.error(f"Binary classification failed: {e}")
            return None
    
    def _run_frame_selector(
        self, 
        signal: np.ndarray, 
        video_path: str
    ) -> Optional[List[List[int]]]:
        """
        Run Frame Selector on video.
        
        Args:
            signal: Binary signal from BC
            video_path: Path to video file
            
        Returns:
            List of [top_indices, bottom_indices], or None on failure
        """
        try:
            capture = cv2.VideoCapture(video_path)
            
            if not capture.isOpened():
                self.logger.error("Failed to open video for FS")
                return None
            
            try:
                indices = self.fs.predict(signal, capture)
                self.logger.debug(
                    f"FS selected: top={len(indices[0])}, bottom={len(indices[1])}"
                )
                return indices
            finally:
                capture.release()
                
        except Exception as e:
            self.logger.error(f"Frame selection failed: {e}")
            return None
    
    def _send_best_frame(
        self,
        frames: List[np.ndarray],
        selected_indices: List[List[int]],
        start_frame: int
    ) -> None:
        """
        Select best frame and send to detection queue.
        
        Prefers top model selection, falls back to bottom model.
        
        Args:
            frames: Original frame list
            selected_indices: [top_indices, bottom_indices] from FS
            start_frame: Starting frame number
        """
        # Select index - prefer top, fallback to bottom
        selected_index = None
        
        if selected_indices[0]:
            selected_index = selected_indices[0][0]
        elif selected_indices[1]:
            selected_index = selected_indices[1][0]
        
        if selected_index is None:
            self.logger.debug("No frame selected by FS")
            return
        
        # Validate index
        if selected_index >= len(frames):
            self.logger.warning(
                f"Selected index {selected_index} out of range ({len(frames)} frames)"
            )
            return
        
        best_frame = frames[selected_index]
        frame_number = start_frame + selected_index
        
        self.logger.info(f"Selected frame {frame_number} for object detection")
        
        try:
            self.detection_queue.put_nowait((best_frame.copy(), frame_number))
        except queue.Full:
            self.logger.warning("Detection queue full - dropping frame")
    
    def stop(self) -> None:
        """
        Signal the thread to stop.
        
        The thread will finish processing the current batch before stopping.
        """
        self.logger.info("Stopping AnalysisThread")
        self.running = False
