"""
Offline Pipeline for batch processing of video files.

This module implements the OfflinePipeline class which processes
pre-recorded video files through the 4-stage crustacean detection pipeline:
Binary Classifier → Frame Selector → Object Detector → Keypoint Detector
"""

import os
import shutil
import time
from pathlib import Path
from typing import Optional, List
import csv

import cv2
import numpy as np

from crustacean.core.pipeline import Pipeline
from crustacean.utils.config import Config
from crustacean.utils.logging_setup import get_logger
from crustacean.utils.exceptions import VideoProcessingError


class OfflinePipeline(Pipeline):
    """
    Batch processing pipeline for pre-recorded video files.
    
    This pipeline processes all video files in a directory through the
    4-stage detection pipeline, saving extracted frames and keypoint
    coordinates to disk.
    
    Attributes:
        video_dir: Directory containing video files to process
        completed_files_path: Path to file tracking completed videos
        extracted_frames_dir: Directory for temporary frame storage
        output_dir: Directory for final output (keypoints CSV)
        profiler: Optional performance profiler
        
    Example:
        >>> config = Config.load()
        >>> pipeline = OfflinePipeline(config, video_dir='./videos')
        >>> pipeline.run()
    """
    
    def __init__(
        self, 
        config: Config, 
        video_dir: str,
        profiler=None
    ):
        """
        Initialize the offline pipeline.
        
        Args:
            config: Configuration object with pipeline settings
            video_dir: Directory containing video files to process
            profiler: Optional PerformanceProfiler instance
        """
        super().__init__(config, profiler=profiler)
        
        self.video_dir = Path(video_dir)
        
        # Get paths from config with defaults
        self.completed_files_path = Path(
            self.config.get('output.completed_files', './CompletedFiles.txt')
        )
        self.extracted_frames_dir = Path(
            self.config.get('output.extracted_frames_dir', './processing/extracted_frames')
        )
        self.output_dir = Path(
            self.config.get('output.detections_dir', './detections')
        )
        
        self.logger.info(f"OfflinePipeline initialized with video_dir={video_dir}")
    
    def run(self) -> None:
        """
        Execute the offline pipeline.
        
        Processes all unprocessed video files in the video directory
        through the 4-stage pipeline.
        """
        self.logger.info("Starting offline pipeline")
        
        # Load completed files list
        completed_files = self._load_completed_files()
        
        # Get list of video files to process
        video_files = self._get_video_files()
        
        if not video_files:
            self.logger.warning(f"No video files found in {self.video_dir}")
            return
        
        # Filter out already completed files
        pending_files = [f for f in video_files if f.name not in completed_files]
        
        if not pending_files:
            self.logger.info("All video files have been processed")
            return
        
        self.logger.info(f"Found {len(pending_files)} videos to process")
        
        # Load models (lazy loading - will load on first use)
        self.load_models(preload=False)
        
        try:
            for video_path in pending_files:
                self._process_video(video_path)
                self._mark_completed(video_path.name)
                
            self.logger.info("Offline pipeline completed successfully")
            
            # Print profiling summary if enabled
            if self.profiler:
                self.profiler.print_summary()
            
        finally:
            self.cleanup()
    
    def _get_video_files(self) -> List[Path]:
        """
        Get list of video files in the video directory.
        
        Returns:
            List of Path objects for video files
        """
        if not self.video_dir.exists():
            self.logger.error(f"Video directory does not exist: {self.video_dir}")
            return []
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        video_files = [
            f for f in self.video_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        
        return sorted(video_files)
    
    def _load_completed_files(self) -> set:
        """
        Load the set of already completed file names.
        
        Returns:
            Set of completed file names
        """
        try:
            if self.completed_files_path.exists():
                with open(self.completed_files_path, 'r') as f:
                    return {line.strip() for line in f if line.strip()}
        except Exception as e:
            self.logger.warning(f"Error loading completed files: {e}")
        
        return set()
    
    def _mark_completed(self, filename: str) -> None:
        """
        Mark a file as completed by adding to the log.
        
        Args:
            filename: Name of the completed file
        """
        try:
            with open(self.completed_files_path, 'a') as f:
                f.write(f"{filename}\n")
        except Exception as e:
            self.logger.warning(f"Error marking file as completed: {e}")
    
    def _process_video(self, video_path: Path) -> None:
        """
        Process a single video through the 4-stage pipeline.
        
        Args:
            video_path: Path to the video file
            
        Raises:
            VideoProcessingError: If processing fails
        """
        self.logger.info(f"Processing video: {video_path.name}")
        start_time = time.time()
        
        try:
            # Stage 1: Binary Classification
            signal = self._run_binary_classifier(video_path)
            
            # Stage 2: Frame Selection
            frame_indices = self._run_frame_selector(signal, video_path)
            del signal  # Free memory
            
            # Stage 3: Object Detection
            roi_frames = self._run_object_detector(video_path, frame_indices)
            del frame_indices  # Free memory
            
            # Stage 4: Keypoint Detection
            keypoints = self._run_keypoint_detector(roi_frames)
            del roi_frames  # Free memory
            
            # Save results
            self._save_results(video_path, keypoints)
            
            elapsed = time.time() - start_time
            self.logger.info(
                f"Finished processing {video_path.name} in {elapsed:.2f}s"
            )
            
        except Exception as e:
            raise VideoProcessingError(
                f"Failed to process video: {video_path.name}",
                details={'video': str(video_path), 'error': str(e)}
            ) from e
    
    def _run_binary_classifier(self, video_path: Path) -> np.ndarray:
        """
        Run binary classifier on video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Binary signal array indicating crustacean presence per frame
        """
        self.logger.debug("Running binary classifier")
        
        bc = self.models['bc']
        if not bc.is_loaded():
            bc.load()
        
        video = cv2.VideoCapture(str(video_path))
        try:
            if self.profiler:
                with self.profiler.profile_section("Binary Classifier"):
                    signal = bc.predict(video)
            else:
                signal = bc.predict(video)
        finally:
            video.release()
        
        bc.unload()  # Free memory
        return signal
    
    def _run_frame_selector(
        self, 
        signal: np.ndarray, 
        video_path: Path
    ) -> List[List[int]]:
        """
        Run frame selector to find best frames.
        
        Args:
            signal: Binary signal from classifier
            video_path: Path to video file
            
        Returns:
            List of [top_indices, bottom_indices]
        """
        self.logger.debug("Running frame selector")
        
        fs = self.models['fs']
        if not fs.is_loaded():
            fs.load()
        
        video = cv2.VideoCapture(str(video_path))
        try:
            if self.profiler:
                with self.profiler.profile_section("Frame Selector"):
                    indices = fs.predict(signal, video)
            else:
                indices = fs.predict(signal, video)
        finally:
            video.release()
        
        fs.unload()  # Free memory
        
        self.logger.debug(
            f"Selected {len(indices[0])} segments "
            f"(top: {len(indices[0])}, bottom: {len(indices[1])})"
        )
        
        return indices
    
    def _run_object_detector(
        self, 
        video_path: Path, 
        frame_indices: List[List[int]]
    ) -> np.ndarray:
        """
        Run object detector on selected frames.
        
        Args:
            video_path: Path to video file
            frame_indices: Indices of frames to process
            
        Returns:
            Array of cropped ROI frames
        """
        self.logger.debug("Running object detector")
        
        # Prepare extracted frames directory
        self._prepare_frames_dir()
        
        # Extract and save selected frames
        self._extract_frames(video_path, frame_indices[0])  # Use top indices
        
        od = self.models['od']
        if not od.is_loaded():
            od.load()
        
        # Process each extracted frame
        roi_frames = []
        frame_files = sorted(self.extracted_frames_dir.glob('*.png'))
        
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                self.logger.warning(f"Failed to read frame: {frame_file}")
                continue
            
            if self.profiler:
                with self.profiler.profile_section("Object Detector"):
                    roi, confidence, class_idx = od.predict(frame)
            else:
                roi, confidence, class_idx = od.predict(frame)
            
            if confidence > 0:
                roi_frames.append(roi)
                self.logger.debug(
                    f"Detection: conf={confidence:.3f}, class={class_idx}"
                )
        
        od.unload()  # Free memory
        
        if not roi_frames:
            self.logger.warning("No valid detections found")
            return np.array([])
        
        return np.array(roi_frames)
    
    def _run_keypoint_detector(self, roi_frames: np.ndarray) -> np.ndarray:
        """
        Run keypoint detector on ROI frames.
        
        Args:
            roi_frames: Array of cropped ROI frames
            
        Returns:
            Array of keypoint coordinates
        """
        if len(roi_frames) == 0:
            self.logger.warning("No ROI frames to process")
            return np.array([])
        
        self.logger.debug(f"Running keypoint detector on {len(roi_frames)} frames")
        
        kd = self.models['kd']
        if not kd.is_loaded():
            kd.load()
        
        if self.profiler:
            with self.profiler.profile_section("Keypoint Detector"):
                keypoints = kd.predict(roi_frames)
        else:
            keypoints = kd.predict(roi_frames)
        
        kd.unload()  # Free memory
        
        self.logger.debug(f"Detected keypoints shape: {keypoints.shape}")
        
        return keypoints
    
    def _prepare_frames_dir(self) -> None:
        """Prepare the extracted frames directory."""
        if self.extracted_frames_dir.exists():
            shutil.rmtree(self.extracted_frames_dir)
        self.extracted_frames_dir.mkdir(parents=True, exist_ok=True)
    
    def _extract_frames(
        self, 
        video_path: Path, 
        frame_indices: List[int]
    ) -> None:
        """
        Extract and save specific frames from video.
        
        Args:
            video_path: Path to video file
            frame_indices: Indices of frames to extract
        """
        video = cv2.VideoCapture(str(video_path))
        
        try:
            for i, frame_idx in enumerate(frame_indices):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = video.read()
                
                if success:
                    output_path = self.extracted_frames_dir / f"{i}.png"
                    cv2.imwrite(str(output_path), frame)
                else:
                    self.logger.warning(f"Failed to extract frame {frame_idx}")
        finally:
            video.release()
    
    def _save_results(self, video_path: Path, keypoints: np.ndarray) -> None:
        """
        Save keypoint results to CSV file.
        
        Args:
            video_path: Path to source video
            keypoints: Array of keypoint coordinates
        """
        if len(keypoints) == 0:
            self.logger.warning("No keypoints to save")
            return
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_file = self.output_dir / f"{video_path.stem}_keypoints.csv"
        
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header: frame_idx, x0, y0, x1, y1, ..., x6, y6
            header = ['frame_idx']
            for i in range(7):
                header.extend([f'x{i}', f'y{i}'])
            writer.writerow(header)
            
            # Data rows
            for i, kp in enumerate(keypoints):
                row = [i] + list(kp)
                writer.writerow(row)
        
        self.logger.info(f"Saved keypoints to {output_file}")
