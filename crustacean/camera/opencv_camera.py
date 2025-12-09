"""
OpenCV camera implementation for USB cameras.

This module provides a camera implementation using standard OpenCV
VideoCapture, suitable for USB webcams and other V4L2 devices.
"""

from typing import Optional
import numpy as np
import cv2

from crustacean.camera.base_camera import BaseCamera
from crustacean.utils.exceptions import CameraInitError
from crustacean.utils.logging_setup import get_logger


class OpenCVCamera(BaseCamera):
    """
    USB camera implementation using OpenCV VideoCapture.
    
    This camera is designed for standard USB webcams and V4L2 devices.
    It uses OpenCV's VideoCapture with device index or path.
    
    Attributes:
        device: Camera device path or index
        capture: OpenCV VideoCapture instance
        
    Configuration options (from config):
        - camera.device: Device path (default: "/dev/video0") or index
        - camera.width: Frame width (default: 1280)
        - camera.height: Frame height (default: 720)
        - camera.framerate: Capture framerate (default: 30)
        
    Example:
        >>> config = Config.load()
        >>> camera = OpenCVCamera(config)
        >>> if camera.open():
        ...     frame = camera.read()
        ...     camera.release()
    """
    
    def __init__(self, config):
        """
        Initialize the OpenCV camera.
        
        Args:
            config: Configuration object with camera settings
        """
        super().__init__(config)
        
        self.device = self._get_device()
        self.capture = None
        
        self.logger.debug(f"OpenCV camera device: {self.device}")
    
    def _get_device(self):
        """
        Get the camera device from configuration.
        
        Returns:
            Device path string or device index integer
        """
        device = self.config.get('camera.device', '/dev/video0')
        
        # If device is a string that looks like an integer, convert it
        if isinstance(device, str):
            if device.isdigit():
                return int(device)
            # Check for common device path patterns
            if device.startswith('/dev/video'):
                return device
        
        return device
    
    def open(self) -> bool:
        """
        Open the USB camera.
        
        Initializes the OpenCV VideoCapture with the configured device
        and sets frame dimensions and framerate.
        
        Returns:
            True if camera opened successfully, False otherwise
            
        Raises:
            CameraInitError: If camera fails to initialize after retries
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Opening OpenCV camera '{self.device}' "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                
                # Open the camera
                if isinstance(self.device, int):
                    self.capture = cv2.VideoCapture(self.device)
                else:
                    self.capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
                
                if not self.capture.isOpened():
                    self.logger.warning(f"Failed to open camera on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay)
                    continue
                
                # Configure camera properties
                self._configure_capture()
                
                self.logger.info("OpenCV camera opened successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Error opening camera: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
        
        self.logger.error("Failed to open OpenCV camera after all retries")
        raise CameraInitError(
            "Failed to open OpenCV camera",
            details={
                'device': str(self.device),
                'attempts': max_retries
            }
        )
    
    def _configure_capture(self) -> None:
        """
        Configure the VideoCapture properties.
        
        Sets frame width, height, and framerate from configuration.
        """
        width = self.config.get('camera.width', 1280)
        height = self.config.get('camera.height', 720)
        framerate = self.config.get('camera.framerate', 30)
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, framerate)
        
        # Log actual values (may differ from requested)
        actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        self.logger.debug(
            f"Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps"
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
            self.logger.info("OpenCV camera released")
    
    def is_opened(self) -> bool:
        """
        Check if the camera is currently open.
        
        Returns:
            True if camera is open and ready, False otherwise
        """
        return self.capture is not None and self.capture.isOpened()
    
    def get_device(self):
        """
        Get the camera device path or index.
        
        Returns:
            Device path string or index integer
        """
        return self.device
