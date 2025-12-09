"""
GStreamer camera implementation for CSI cameras.

This module provides a camera implementation using GStreamer pipelines,
primarily designed for NVIDIA Jetson devices with CSI cameras.
"""

from typing import Optional
import numpy as np
import cv2

from crustacean.camera.base_camera import BaseCamera
from crustacean.utils.exceptions import CameraInitError
from crustacean.utils.logging_setup import get_logger


class GStreamerCamera(BaseCamera):
    """
    CSI camera implementation using GStreamer pipeline.
    
    This camera is designed for NVIDIA Jetson devices using CSI cameras
    with the nvarguscamerasrc GStreamer element. It builds a configurable
    pipeline based on the provided configuration.
    
    Attributes:
        pipeline: GStreamer pipeline string
        capture: OpenCV VideoCapture instance
        
    Configuration options (from config):
        - camera.width: Frame width (default: 1280)
        - camera.height: Frame height (default: 720)
        - camera.framerate: Capture framerate (default: 45)
        - camera.rotation: Rotation angle in degrees (default: 180)
        
    Example:
        >>> config = Config.load()
        >>> camera = GStreamerCamera(config)
        >>> if camera.open():
        ...     frame = camera.read()
        ...     camera.release()
    """
    
    def __init__(self, config):
        """
        Initialize the GStreamer camera.
        
        Args:
            config: Configuration object with camera settings
        """
        super().__init__(config)
        
        self.pipeline = self._build_pipeline()
        self.capture = None
        
        self.logger.debug(f"GStreamer pipeline: {self.pipeline}")
    
    def _build_pipeline(self) -> str:
        """
        Build the GStreamer pipeline string from configuration.
        
        The pipeline uses nvarguscamerasrc for CSI camera capture,
        nvvidconv for format conversion, and videoflip for rotation.
        
        Returns:
            GStreamer pipeline string
        """
        width = self.config.get('camera.width', 1280)
        height = self.config.get('camera.height', 720)
        framerate = self.config.get('camera.framerate', 45)
        rotation = self.config.get('camera.rotation', 180)
        
        # Map rotation degrees to GStreamer videoflip method
        rotation_method = self._get_rotation_method(rotation)
        
        pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},"
            f"framerate={framerate}/1 ! "
            f"nvvidconv ! "
            f"videoflip method={rotation_method} ! "
            f"video/x-raw,format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=2 sync=false"
        )
        
        return pipeline
    
    def _get_rotation_method(self, degrees: int) -> str:
        """
        Convert rotation degrees to GStreamer videoflip method.
        
        Args:
            degrees: Rotation angle (0, 90, 180, 270)
            
        Returns:
            GStreamer videoflip method string
        """
        rotation_map = {
            0: "none",
            90: "clockwise",
            180: "rotate-180",
            270: "counterclockwise"
        }
        return rotation_map.get(degrees, "rotate-180")
    
    def open(self) -> bool:
        """
        Open the GStreamer camera.
        
        Initializes the OpenCV VideoCapture with the GStreamer pipeline.
        
        Returns:
            True if camera opened successfully, False otherwise
            
        Raises:
            CameraInitError: If camera fails to initialize after retries
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Opening GStreamer camera (attempt {attempt + 1}/{max_retries})")
                
                self.capture = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
                
                if self.capture.isOpened():
                    self.logger.info("GStreamer camera opened successfully")
                    return True
                
                self.logger.warning(f"Failed to open camera on attempt {attempt + 1}")
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    
            except Exception as e:
                self.logger.error(f"Error opening camera: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
        
        self.logger.error("Failed to open GStreamer camera after all retries")
        raise CameraInitError(
            "Failed to open GStreamer camera",
            details={
                'pipeline': self.pipeline,
                'attempts': max_retries
            }
        )
    
    def read(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the camera.
        
        Returns:
            BGR frame as numpy array, or None if read failed
        """
        if self.capture is None or not self.capture.isOpened():
            self.logger.warning("Attempted to read from closed camera")
            return None
        
        ret, frame = self.capture.read()
        
        if not ret:
            self.logger.warning("Failed to read frame from camera")
            return None
        
        return frame
    
    def release(self) -> None:
        """
        Release the camera resources.
        
        Closes the VideoCapture and frees associated resources.
        Safe to call multiple times.
        """
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self.logger.info("GStreamer camera released")
    
    def is_opened(self) -> bool:
        """
        Check if the camera is currently open.
        
        Returns:
            True if camera is open and ready, False otherwise
        """
        return self.capture is not None and self.capture.isOpened()
    
    def get_pipeline(self) -> str:
        """
        Get the GStreamer pipeline string.
        
        Returns:
            The GStreamer pipeline string being used
        """
        return self.pipeline
