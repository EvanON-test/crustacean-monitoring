"""
Base camera interface for the Crustacean Monitoring System.

This module defines the abstract base class for all camera implementations,
providing a consistent interface for camera operations across different
backends (GStreamer, OpenCV, RTSP, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

from crustacean.utils.logging_setup import get_logger


class BaseCamera(ABC):
    """
    Abstract base class for camera interfaces.
    
    This class defines the contract that all camera implementations must follow,
    ensuring consistent behavior across different camera backends.
    
    Subclasses must implement:
        - open(): Initialize and open the camera connection
        - read(): Capture and return the next frame
        - release(): Close the camera and release resources
        - is_opened(): Check if the camera is currently open
    
    Attributes:
        config: Configuration object with camera settings
        logger: Logger instance for this camera
        
    Example:
        >>> class MyCamera(BaseCamera):
        ...     def open(self) -> bool:
        ...         # Implementation
        ...         return True
        ...     def read(self) -> Optional[np.ndarray]:
        ...         # Implementation
        ...         return frame
        ...     def release(self) -> None:
        ...         # Implementation
        ...         pass
        ...     def is_opened(self) -> bool:
        ...         return True
    """
    
    def __init__(self, config):
        """
        Initialize the base camera.
        
        Args:
            config: Configuration object with camera settings
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def open(self) -> bool:
        """
        Open the camera connection.
        
        This method should initialize the camera hardware/software and
        prepare it for frame capture.
        
        Returns:
            True if camera opened successfully, False otherwise
            
        Raises:
            CameraInitError: If camera initialization fails critically
        """
        pass
    
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the camera.
        
        Returns:
            numpy.ndarray: BGR frame if successful, None if read failed
            
        Note:
            The returned frame should be in BGR format (OpenCV standard)
            with shape (height, width, 3) and dtype uint8.
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """
        Release camera resources.
        
        This method should close the camera connection and free any
        associated resources. It should be safe to call multiple times.
        """
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check if the camera is currently open and ready.
        
        Returns:
            True if camera is open and ready for capture, False otherwise
        """
        pass
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the configured frame size.
        
        Returns:
            Tuple of (width, height) from configuration
        """
        width = self.config.get('camera.width', 1280)
        height = self.config.get('camera.height', 720)
        return (width, height)
    
    def get_framerate(self) -> int:
        """
        Get the configured framerate.
        
        Returns:
            Framerate in frames per second
        """
        return self.config.get('camera.framerate', 30)
    
    def __enter__(self):
        """
        Context manager entry - opens the camera.
        
        Returns:
            self if camera opened successfully
            
        Raises:
            CameraInitError: If camera fails to open
        """
        from crustacean.utils.exceptions import CameraInitError
        
        if not self.open():
            raise CameraInitError(
                f"Failed to open {self.__class__.__name__}",
                details={'camera_type': self.__class__.__name__}
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - releases the camera.
        """
        self.release()
        return False  # Don't suppress exceptions
    
    def __repr__(self) -> str:
        """String representation of the camera."""
        status = "opened" if self.is_opened() else "closed"
        width, height = self.get_frame_size()
        return f"{self.__class__.__name__}({width}x{height}, {status})"
