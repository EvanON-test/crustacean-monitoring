"""
Camera interface abstractions for the Crustacean Monitoring System.

This package provides camera interfaces for different backends including:
- GStreamer (CSI cameras on NVIDIA Jetson)
- OpenCV (USB cameras and V4L2 devices)

The factory function `create_camera()` automatically selects the appropriate
camera implementation based on configuration.

Example:
    >>> from crustacean.camera import create_camera
    >>> from crustacean.utils.config import Config
    >>> 
    >>> config = Config.load()
    >>> camera = create_camera(config)
    >>> 
    >>> with camera:
    ...     frame = camera.read()
    ...     # Process frame
"""

from crustacean.camera.base_camera import BaseCamera
from crustacean.camera.gstreamer_camera import GStreamerCamera
from crustacean.camera.opencv_camera import OpenCVCamera


def create_camera(config) -> BaseCamera:
    """
    Factory function to create appropriate camera based on configuration.
    
    This function reads the camera type from configuration and returns
    the appropriate camera implementation.
    
    Supported camera types:
        - 'csi': GStreamer camera for CSI cameras (NVIDIA Jetson)
        - 'usb': OpenCV camera for USB webcams
        - 'gstreamer': Alias for 'csi'
        - 'opencv': Alias for 'usb'
    
    Args:
        config: Configuration object with camera settings
            Required config keys:
            - camera.type: Camera type ('csi', 'usb', 'gstreamer', 'opencv')
            
    Returns:
        Camera instance (GStreamerCamera or OpenCVCamera)
        
    Raises:
        ConfigurationError: If camera type is invalid or unsupported
        
    Example:
        >>> config = Config.load()
        >>> camera = create_camera(config)
        >>> if camera.open():
        ...     frame = camera.read()
        ...     camera.release()
    """
    from crustacean.utils.exceptions import ConfigurationError
    from crustacean.utils.logging_setup import get_logger
    
    logger = get_logger(__name__)
    
    camera_type = config.get('camera.type', 'csi').lower()
    
    logger.info(f"Creating camera of type: {camera_type}")
    
    # Map camera types to implementations
    camera_map = {
        'csi': GStreamerCamera,
        'gstreamer': GStreamerCamera,
        'usb': OpenCVCamera,
        'opencv': OpenCVCamera,
    }
    
    if camera_type not in camera_map:
        supported = ', '.join(sorted(camera_map.keys()))
        raise ConfigurationError(
            f"Unsupported camera type: '{camera_type}'",
            details={
                'camera_type': camera_type,
                'supported_types': supported
            }
        )
    
    camera_class = camera_map[camera_type]
    return camera_class(config)


__all__ = [
    "BaseCamera",
    "GStreamerCamera",
    "OpenCVCamera",
    "create_camera",
]
