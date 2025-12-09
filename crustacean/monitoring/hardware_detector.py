"""
Hardware detection and monitor factory.

This module provides automatic hardware platform detection and
creates the appropriate monitor instance for the detected platform.
"""

import os
import platform
from typing import TYPE_CHECKING

from crustacean.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from crustacean.utils.config import Config
    from crustacean.monitoring.base_monitor import BaseMonitor

logger = get_logger(__name__)


def detect_hardware() -> str:
    """
    Auto-detect the hardware platform.
    
    Checks system characteristics to determine if running on:
    - Jetson (NVIDIA Jetson Nano, TX2, Xavier, etc.)
    - Raspberry Pi
    - Generic x86/x64 system
    
    Returns:
        str: One of 'jetson', 'raspberry_pi', or 'generic'
        
    Example:
        >>> hardware = detect_hardware()
        >>> print(f"Running on: {hardware}")
        Running on: jetson
    """
    machine = platform.machine()
    system = platform.system()
    
    logger.debug(f"Detecting hardware: machine={machine}, system={system}")
    
    # Check for Jetson (aarch64 with NVIDIA Tegra)
    if machine == "aarch64":
        # Check for Jetson-specific files
        if os.path.exists('/etc/nv_tegra_release'):
            logger.info("Detected NVIDIA Jetson platform")
            return 'jetson'
        
        # Alternative check via /proc/device-tree
        if os.path.exists('/proc/device-tree/model'):
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    if 'nvidia' in model or 'jetson' in model:
                        logger.info("Detected NVIDIA Jetson platform (via device-tree)")
                        return 'jetson'
            except (IOError, OSError):
                pass
    
    # Check for Raspberry Pi
    if machine in ("armv7l", "armv6l", "aarch64"):
        # Check for Raspberry Pi specific files
        if os.path.exists('/proc/device-tree/model'):
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    if 'raspberry' in model:
                        logger.info("Detected Raspberry Pi platform")
                        return 'raspberry_pi'
            except (IOError, OSError):
                pass
        
        # Alternative check via /proc/cpuinfo
        if os.path.exists('/proc/cpuinfo'):
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'raspberry' in cpuinfo or 'bcm' in cpuinfo:
                        logger.info("Detected Raspberry Pi platform (via cpuinfo)")
                        return 'raspberry_pi'
            except (IOError, OSError):
                pass
    
    logger.info("Detected generic platform")
    return 'generic'


def create_monitor(config: 'Config', output_file: str = None) -> 'BaseMonitor':
    """
    Factory function to create the appropriate monitor for the platform.
    
    Automatically detects the hardware platform and creates a monitor
    instance with platform-specific metric collection capabilities.
    
    Args:
        config: Configuration object with monitoring settings
        output_file: Path to output CSV file for metrics.
                    If None, uses config default or 'metrics.csv'
    
    Returns:
        BaseMonitor: Platform-specific monitor instance
        
    Example:
        >>> config = Config.load()
        >>> monitor = create_monitor(config, 'hardware_metrics.csv')
        >>> monitor.start()
        >>> # ... run pipeline ...
        >>> monitor.stop()
    """
    # Import here to avoid circular imports
    from crustacean.monitoring.jetson_monitor import JetsonMonitor
    from crustacean.monitoring.pi_monitor import RaspberryPiMonitor
    from crustacean.monitoring.generic_monitor import GenericMonitor
    
    # Determine output file
    if output_file is None:
        output_file = config.get('monitoring.output_file', 'metrics.csv')
    
    # Detect hardware and create appropriate monitor
    hardware = detect_hardware()
    
    if hardware == 'jetson':
        logger.info("Creating JetsonMonitor")
        return JetsonMonitor(config, output_file)
    elif hardware == 'raspberry_pi':
        logger.info("Creating RaspberryPiMonitor")
        return RaspberryPiMonitor(config, output_file)
    else:
        logger.info("Creating GenericMonitor")
        return GenericMonitor(config, output_file)
