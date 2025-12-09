"""
Hardware monitoring and metrics collection.

This package provides platform-specific hardware monitoring capabilities
with automatic hardware detection and graceful fallbacks.

Supported Platforms:
    - NVIDIA Jetson (Nano, TX2, Xavier, Orin) via jtop
    - Raspberry Pi via gpiozero
    - Generic x86/x64 systems via psutil

Usage:
    >>> from crustacean.monitoring import create_monitor, detect_hardware
    >>> 
    >>> # Auto-detect platform and create appropriate monitor
    >>> monitor = create_monitor(config, 'metrics.csv')
    >>> monitor.start()
    >>> 
    >>> # ... run pipeline ...
    >>> 
    >>> monitor.stop()
    >>> monitor.join()
"""

from crustacean.monitoring.base_monitor import BaseMonitor
from crustacean.monitoring.hardware_detector import detect_hardware, create_monitor
from crustacean.monitoring.jetson_monitor import JetsonMonitor
from crustacean.monitoring.pi_monitor import RaspberryPiMonitor
from crustacean.monitoring.generic_monitor import GenericMonitor

__all__ = [
    "BaseMonitor",
    "detect_hardware",
    "create_monitor",
    "JetsonMonitor",
    "RaspberryPiMonitor",
    "GenericMonitor",
]
