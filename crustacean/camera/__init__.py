"""
Camera interface abstractions.

This package provides camera interfaces for different backends including
GStreamer (CSI cameras), OpenCV (USB cameras), and RTSP streams.
"""

from crustacean.camera.base_camera import BaseCamera
from crustacean.camera.gstreamer_camera import GStreamerCamera
from crustacean.camera.opencv_camera import OpenCVCamera

def create_camera(config):
    """
    Factory function to create appropriate camera based on configuration.
    
    Args:
        config: Configuration object with camera settings
        
    Returns:
        Camera instance (GStreamerCamera, OpenCVCamera, etc.)
        
    Raises:
        ConfigurationError: If camera type is invalid
    """
    from crustacean.utils.exceptions import ConfigurationError
    
    camera_type = config.get('camera.type', 'csi')
    
    if camera_type == 'csi':
        return GStreamerCamera(config)
    elif camera_type == 'usb':
        return OpenCVCamera(config)
    else:
        raise ConfigurationError(f"Unsupported camera type: {camera_type}")

__all__ = [
    "BaseCamera",
    "GStreamerCamera",
    "OpenCVCamera",
    "create_camera",
]
