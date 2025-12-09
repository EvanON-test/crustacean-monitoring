"""
Real-time Pipeline for live camera processing.

This module implements the RealtimePipeline class which processes
live camera feeds through the crustacean detection pipeline using
multi-threaded architecture for optimal performance.
"""

import gc
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from crustacean.core.pipeline import Pipeline
from crustacean.camera import create_camera, BaseCamera
from crustacean.threads import AnalysisThread, DetectionThread, save_detection
from crustacean.utils.config import Config
from crustacean.utils.logging_setup import get_logger
from crustacean.utils.exceptions import CameraInitError


class RealtimePipeline(Pipeline):
    """
    Live camera processing pipeline with multi-threading.
    
    This pipeline captures frames from a camera, detects motion,
    and processes detected motion through the 4-stage crustacean
    detection pipeline using background threads.
    
    Architecture:
        MainThread: Camera capture, motion detection, orchestration
        AnalysisThread: Binary classification and frame selection
        DetectionThread: Object detection
        ThreadPoolExecutor: Parallel save operations
    
    Attributes:
        display_mode: Whether to show video display
        camera: Camera instance for frame capture
        threads: Dictionary of background threads
        queues: Dictionary of inter-thread queues
        executor: ThreadPoolExecutor for save operations
        
    Example:
        >>> config = Config.load()
        >>> pipeline = RealtimePipeline(config, display_mode=False)
        >>> pipeline.run()
    """
    
    def __init__(self, config: Config, display_mode: bool = False, profiler=None):
        """
        Initialize the real-time pipeline.
        
        Args:
            config: Configuration object with pipeline settings
            display_mode: If True, display video with overlays
            profiler: Optional PerformanceProfiler for timing measurements
        """
        super().__init__(config, profiler=profiler)
        
        self.display_mode = display_mode
        self.camera: Optional[BaseCamera] = None
        self.threads: Dict[str, Any] = {}
        self.queues: Dict[str, queue.Queue] = {}
        self.executor: Optional[ThreadPoolExecutor] = None
        
        # Motion detection state
        self.previous_frame: Optional[np.ndarray] = None
        self.motion_threshold = self.config.get('realtime.motion_detection_threshold', 15)
        
        # Frame collection state
        self.collecting = False
        self.collected_frames: List[np.ndarray] = []
        self.collect_start_frame = 0
        self.frames_to_collect = self.config.get('realtime.frames_to_collect', 30)
        
        # Processing intervals
        self.process_interval = self.config.get('realtime.process_interval', 30)
        
        # Detection cooldown
        self.last_detection_time = 0.0
        self.detection_cooldown = self.config.get('realtime.detection_cooldown', 3)
        
        # Statistics
        self.frame_counter = 0
        self.detection_count = 0
        self.start_time = 0.0
        
        # Display state
        self.latest_confidence = 0.0
        
        self.logger.info(
            f"RealtimePipeline initialized (display_mode={display_mode})"
        )
    
    def run(self) -> None:
        """
        Execute the real-time pipeline.
        
        Initializes all components, runs the main processing loop,
        and handles graceful shutdown.
        """
        self.start_time = time.time()
        
        try:
            self._initialize()
            self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
            
        except Exception as e:
            self.logger.exception(f"Pipeline error: {e}")
            
        finally:
            self._shutdown()
    
    def _initialize(self) -> None:
        """
        Initialize camera, models, threads, and executor.
        
        Raises:
            CameraInitError: If camera fails to initialize
        """
        self.logger.info("Initializing real-time pipeline")
        
        # Initialize camera
        self._initialize_camera()
        
        # Load models (preload for real-time performance)
        self.load_models(preload=True)
        
        # Initialize queues
        self._initialize_queues()
        
        # Start background threads
        self._start_threads()
        
        # Initialize thread pool for save operations
        max_save_threads = self.config.get('realtime.max_save_threads', 2)
        self.executor = ThreadPoolExecutor(max_workers=max_save_threads)
        
        self.logger.info("Real-time pipeline initialized successfully")
    
    def _initialize_camera(self) -> None:
        """
        Initialize and open the camera.
        
        Raises:
            CameraInitError: If camera fails to open
        """
        self.logger.info("Initializing camera")
        
        self.camera = create_camera(self.config)
        
        if not self.camera.open():
            raise CameraInitError(
                "Failed to open camera",
                details={'camera_type': type(self.camera).__name__}
            )
        
        self.logger.info(f"Camera opened: {self.camera}")
    
    def _initialize_queues(self) -> None:
        """Initialize inter-thread communication queues."""
        self.queues['analysis'] = queue.Queue(maxsize=1)
        self.queues['detection'] = queue.Queue(maxsize=1)
        self.queues['results'] = queue.Queue(maxsize=1)
        
        self.logger.debug("Queues initialized")
    
    def _start_threads(self) -> None:
        """Start background processing threads."""
        self.logger.info("Starting background threads")
        
        # Analysis thread (BC + FS)
        self.threads['analysis'] = AnalysisThread(
            analysis_queue=self.queues['analysis'],
            detection_queue=self.queues['detection'],
            bc_model=self.models['bc'],
            fs_model=self.models['fs'],
            config=self.config
        )
        self.threads['analysis'].start()
        
        # Detection thread (OD)
        self.threads['detection'] = DetectionThread(
            frame_queue=self.queues['detection'],
            result_queue=self.queues['results'],
            od_model=self.models['od'],
            config=self.config
        )
        self.threads['detection'].start()
        
        self.logger.info("Background threads started")

    
    def _main_loop(self) -> None:
        """
        Main processing loop.
        
        Captures frames, detects motion, collects frames for analysis,
        and handles detection results.
        """
        self.logger.info("Starting main processing loop")
        self.logger.info(f"Processing every {self.process_interval} frames")
        self.logger.info("Press Ctrl+C to stop")
        
        while True:
            # Capture frame
            if self.profiler:
                with self.profiler.profile_section("Frame Capture"):
                    frame = self.camera.read()
            else:
                frame = self.camera.read()
            
            if frame is None:
                self.logger.warning("Failed to capture frame")
                continue
            
            self.frame_counter += 1
            
            # Check for motion at intervals
            if self._should_process_frame(self.frame_counter):
                self._check_motion(frame)
            
            # Collect frames if triggered
            if self.collecting:
                self._collect_frame(frame)
            
            # Handle detection results
            self._handle_detection_results()
            
            # Display frame if in display mode
            if self.display_mode:
                if not self._render_frame(frame):
                    break  # User requested quit
    
    def _should_process_frame(self, frame_number: int) -> bool:
        """
        Check if this frame should be processed for motion.
        
        Args:
            frame_number: Current frame number
            
        Returns:
            True if frame should be processed
        """
        return frame_number % self.process_interval == 0
    
    def _check_motion(self, frame: np.ndarray) -> None:
        """
        Check for motion and start collection if detected.
        
        Args:
            frame: Current frame to check
        """
        current_time = time.time()
        time_since_detection = current_time - self.last_detection_time
        
        # Check cooldown
        if time_since_detection <= self.detection_cooldown:
            self.logger.debug("In detection cooldown period")
            return
        
        # Detect motion
        if self._detect_motion(frame):
            if not self.collecting:
                self.logger.info("Motion detected - starting frame collection")
                self.collecting = True
                self.collected_frames = []
                self.collect_start_frame = self.frame_counter
                self.last_detection_time = current_time
    
    def _detect_motion(self, frame: np.ndarray) -> bool:
        """
        Detect motion using frame differencing.
        
        Compares the current frame with the previous frame to detect
        significant changes indicating motion.
        
        Args:
            frame: Current BGR frame
            
        Returns:
            True if motion detected above threshold
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize previous frame
        if self.previous_frame is None:
            self.previous_frame = gray
            return False
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        self.previous_frame = gray
        
        # Apply threshold
        _, thresh = cv2.threshold(
            frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY
        )
        
        # Calculate movement percentage
        non_zero = cv2.countNonZero(thresh)
        total_pixels = thresh.size
        movement_percent = (non_zero / total_pixels) * 100
        
        motion_detected = movement_percent > self.motion_threshold
        
        if motion_detected:
            self.logger.debug(f"Motion: {movement_percent:.2f}%")
        
        return motion_detected
    
    def _collect_frame(self, frame: np.ndarray) -> None:
        """
        Collect frame for analysis batch.
        
        Args:
            frame: Frame to add to collection
        """
        self.collected_frames.append(frame.copy())
        
        if len(self.collected_frames) >= self.frames_to_collect:
            self.logger.info(
                f"Collection complete: {len(self.collected_frames)} frames"
            )
            self._submit_for_analysis()
            self.collecting = False
            self.collected_frames = []
    
    def _submit_for_analysis(self) -> None:
        """Submit collected frames to analysis thread."""
        try:
            self.queues['analysis'].put_nowait(
                (self.collected_frames.copy(), self.collect_start_frame)
            )
            self.logger.debug("Frames submitted for analysis")
        except queue.Full:
            self.logger.warning("Analysis queue full - dropping frames")
    
    def _handle_detection_results(self) -> None:
        """Process detection results from the detection thread."""
        try:
            while not self.queues['results'].empty():
                result = self.queues['results'].get_nowait()
                
                self.logger.info(
                    f"Detection result: frame={result.frame_number}, "
                    f"confidence={result.confidence:.3f}"
                )
                
                self.latest_confidence = result.confidence
                
                # Check confidence threshold
                confidence_threshold = self.config.get(
                    'models.object_detector.confidence_threshold', 0.75
                )
                
                if result.confidence >= confidence_threshold:
                    self.detection_count += 1
                    self._save_detection(result)
                else:
                    self.logger.debug(
                        f"Low confidence detection: {result.confidence:.3f}"
                    )
                
                # Clean up
                del result
                gc.collect()
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error handling detection results: {e}")
    
    def _save_detection(self, result) -> None:
        """
        Submit detection for saving via thread pool.
        
        Args:
            result: DetectionResult to save
        """
        try:
            self.executor.submit(
                save_detection,
                result.frame,
                result.roi,
                result.confidence,
                result.frame_number,
                self.config,
                self.models.get('kd')
            )
            self.logger.info(f"Detection {self.detection_count} submitted for saving")
        except Exception as e:
            self.logger.error(f"Failed to submit save task: {e}")
    
    def _render_frame(self, frame: np.ndarray) -> bool:
        """
        Render frame with overlays (display mode only).
        
        Args:
            frame: Frame to display
            
        Returns:
            True to continue, False to quit
        """
        display_frame = frame.copy()
        
        # Add status overlay
        self._draw_overlay(display_frame)
        
        # Show frame
        cv2.imshow('Crustacean Monitor', display_frame)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.logger.info("Quit requested via keyboard")
            return False
        
        return True
    
    def _draw_overlay(self, frame: np.ndarray) -> None:
        """
        Draw status overlay on frame.
        
        Args:
            frame: Frame to draw on (modified in place)
        """
        # Status text
        runtime = time.time() - self.start_time
        status_lines = [
            f"Frame: {self.frame_counter}",
            f"Detections: {self.detection_count}",
            f"Runtime: {runtime:.1f}s",
            f"Confidence: {self.latest_confidence:.2f}",
        ]
        
        if self.collecting:
            status_lines.append(f"Collecting: {len(self.collected_frames)}/{self.frames_to_collect}")
        
        # Draw text
        y_offset = 30
        for line in status_lines:
            cv2.putText(
                frame, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            y_offset += 30
    
    def _shutdown(self) -> None:
        """
        Graceful shutdown of all components.
        
        Stops threads, waits for saves to complete, releases camera,
        and unloads models.
        """
        self.logger.info("Shutting down real-time pipeline")
        shutdown_start = time.time()
        
        # Stop background threads
        self._stop_threads()
        
        # Wait for save operations to complete
        self._shutdown_executor()
        
        # Release camera
        self._release_camera()
        
        # Close display window
        if self.display_mode:
            cv2.destroyAllWindows()
        
        # Unload models
        self.cleanup()
        
        # Log summary
        self._log_summary(shutdown_start)
        
        # Print profiling summary if enabled
        if self.profiler:
            self.profiler.print_summary()
    
    def _stop_threads(self) -> None:
        """Stop all background threads."""
        self.logger.info("Stopping background threads")
        
        for name, thread in self.threads.items():
            try:
                thread.stop()
                thread.join(timeout=5)
                
                if thread.is_alive():
                    self.logger.warning(f"Thread {name} did not stop gracefully")
                else:
                    self.logger.debug(f"Thread {name} stopped")
                    
            except Exception as e:
                self.logger.error(f"Error stopping thread {name}: {e}")
        
        self.threads.clear()
    
    def _shutdown_executor(self) -> None:
        """Shutdown the thread pool executor."""
        if self.executor:
            self.logger.info("Waiting for save operations to complete")
            try:
                self.executor.shutdown(wait=True)
                self.logger.debug("Executor shutdown complete")
            except Exception as e:
                self.logger.error(f"Error shutting down executor: {e}")
            self.executor = None
    
    def _release_camera(self) -> None:
        """Release camera resources."""
        if self.camera:
            try:
                self.camera.release()
                self.logger.debug("Camera released")
            except Exception as e:
                self.logger.error(f"Error releasing camera: {e}")
            self.camera = None
    
    def _log_summary(self, shutdown_start: float) -> None:
        """
        Log pipeline run summary.
        
        Args:
            shutdown_start: Time when shutdown started
        """
        runtime = time.time() - self.start_time
        shutdown_time = time.time() - shutdown_start
        
        self.logger.info("=" * 50)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info(f"  Total frames processed: {self.frame_counter}")
        self.logger.info(f"  High-confidence detections: {self.detection_count}")
        self.logger.info(f"  Total runtime: {runtime:.2f}s")
        self.logger.info(f"  Shutdown time: {shutdown_time:.2f}s")
        self.logger.info("=" * 50)
