"""
Raspberry Pi hardware monitor.

This module provides hardware monitoring specific to Raspberry Pi
platforms using gpiozero for CPU temperature readings.
"""

from typing import Dict, Any, List, TYPE_CHECKING

from crustacean.monitoring.base_monitor import BaseMonitor
from crustacean.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from crustacean.utils.config import Config

# Try to import gpiozero - may not be available on non-Pi platforms
try:
    from gpiozero import CPUTemperature
    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False


class RaspberryPiMonitor(BaseMonitor):
    """
    Hardware monitor for Raspberry Pi platforms.
    
    Uses gpiozero to collect Pi-specific metrics including:
    - CPU temperature
    
    Falls back to common metrics if gpiozero is not available.
    
    Attributes:
        cpu_temp: CPUTemperature instance for reading temperature
        
    Example:
        >>> monitor = RaspberryPiMonitor(config, 'pi_metrics.csv')
        >>> monitor.start()
        >>> # ... run pipeline ...
        >>> monitor.stop()
        >>> monitor.join()
    """
    
    def __init__(self, config: 'Config', output_file: str):
        """
        Initialize the Raspberry Pi monitor.
        
        Args:
            config: Configuration object with monitoring settings
            output_file: Path to output CSV file for metrics
        """
        super().__init__(config, output_file)
        
        self.cpu_temp = None
        
        if GPIOZERO_AVAILABLE:
            try:
                self.cpu_temp = CPUTemperature()
                self.logger.info("gpiozero CPUTemperature initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CPUTemperature: {e}")
        else:
            self.logger.warning(
                "gpiozero not available - CPU temperature will be unavailable. "
                "Install with: pip install gpiozero"
            )
    
    def get_platform_fieldnames(self) -> List[str]:
        """
        Get Raspberry Pi-specific CSV field names.
        
        Returns:
            List of Pi-specific field names
        """
        return ['cpu_temp']
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect all hardware metrics including Pi-specific ones.
        
        Returns:
            Dictionary with all collected metrics
        """
        # Start with common metrics
        metrics = self.get_common_metrics()
        
        # Add Pi-specific metrics
        if self.cpu_temp is not None:
            try:
                metrics['cpu_temp'] = round(self.cpu_temp.temperature, 1)
            except Exception as e:
                self.logger.warning(f"Error reading CPU temperature: {e}")
                metrics['cpu_temp'] = 'N/A'
        else:
            # Try reading from /sys/class/thermal as fallback
            metrics['cpu_temp'] = self._read_thermal_zone()
        
        return metrics
    
    def _read_thermal_zone(self) -> Any:
        """
        Read CPU temperature from thermal zone file.
        
        Fallback method when gpiozero is not available.
        
        Returns:
            Temperature in Celsius or 'N/A' if unavailable
        """
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millidegrees = int(f.read().strip())
                return round(temp_millidegrees / 1000.0, 1)
        except (IOError, ValueError, OSError):
            return 'N/A'
