"""
Generic hardware monitor.

This module provides hardware monitoring for generic platforms
(x86/x64 systems) using only psutil for cross-platform compatibility.
"""

from typing import Dict, Any, List, TYPE_CHECKING

import psutil

from crustacean.monitoring.base_monitor import BaseMonitor
from crustacean.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from crustacean.utils.config import Config


class GenericMonitor(BaseMonitor):
    """
    Hardware monitor for generic platforms.
    
    Uses psutil to collect cross-platform metrics including:
    - CPU usage per core
    - Disk I/O statistics
    - Network I/O statistics
    
    This monitor works on any platform supported by psutil.
    
    Example:
        >>> monitor = GenericMonitor(config, 'metrics.csv')
        >>> monitor.start()
        >>> # ... run pipeline ...
        >>> monitor.stop()
        >>> monitor.join()
    """
    
    def __init__(self, config: 'Config', output_file: str):
        """
        Initialize the generic monitor.
        
        Args:
            config: Configuration object with monitoring settings
            output_file: Path to output CSV file for metrics
        """
        super().__init__(config, output_file)
        
        # Initialize baseline counters for delta calculations
        self._last_disk_io = psutil.disk_io_counters()
        self._last_net_io = psutil.net_io_counters()
        
        self.logger.info("Generic monitor initialized")
    
    def get_platform_fieldnames(self) -> List[str]:
        """
        Get generic platform CSV field names.
        
        Returns:
            List of generic platform field names
        """
        fields = ['cpu_temp']
        
        # Add per-core CPU fields
        cpu_count = psutil.cpu_count()
        if cpu_count:
            for i in range(min(cpu_count, 8)):  # Limit to 8 cores
                fields.append(f'cpu{i}_percent')
        
        # Add I/O fields
        fields.extend([
            'disk_read_mb',
            'disk_write_mb',
            'net_sent_mb',
            'net_recv_mb'
        ])
        
        return fields
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect all hardware metrics.
        
        Returns:
            Dictionary with all collected metrics
        """
        # Start with common metrics
        metrics = self.get_common_metrics()
        
        # CPU temperature (if available)
        metrics['cpu_temp'] = self._get_cpu_temperature()
        
        # Per-core CPU usage
        try:
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            for i, percent in enumerate(per_cpu[:8]):  # Limit to 8 cores
                metrics[f'cpu{i}_percent'] = percent
        except Exception as e:
            self.logger.warning(f"Error getting per-CPU stats: {e}")
        
        # Disk I/O
        disk_metrics = self._get_disk_io_metrics()
        metrics.update(disk_metrics)
        
        # Network I/O
        net_metrics = self._get_network_io_metrics()
        metrics.update(net_metrics)
        
        return metrics
    
    def _get_cpu_temperature(self) -> Any:
        """
        Get CPU temperature using psutil sensors.
        
        Returns:
            Temperature in Celsius or 'N/A' if unavailable
        """
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try common sensor names
                    for name in ['coretemp', 'cpu_thermal', 'k10temp', 'acpitz']:
                        if name in temps and temps[name]:
                            return round(temps[name][0].current, 1)
                    # Return first available sensor
                    for name, entries in temps.items():
                        if entries:
                            return round(entries[0].current, 1)
        except Exception:
            pass
        
        return 'N/A'
    
    def _get_disk_io_metrics(self) -> Dict[str, Any]:
        """
        Get disk I/O metrics (delta since last collection).
        
        Returns:
            Dictionary with disk I/O metrics in MB
        """
        metrics = {
            'disk_read_mb': 0.0,
            'disk_write_mb': 0.0
        }
        
        try:
            current = psutil.disk_io_counters()
            if current and self._last_disk_io:
                # Calculate delta
                read_bytes = current.read_bytes - self._last_disk_io.read_bytes
                write_bytes = current.write_bytes - self._last_disk_io.write_bytes
                
                metrics['disk_read_mb'] = round(read_bytes / (1024 * 1024), 2)
                metrics['disk_write_mb'] = round(write_bytes / (1024 * 1024), 2)
                
            self._last_disk_io = current
        except Exception as e:
            self.logger.warning(f"Error getting disk I/O: {e}")
        
        return metrics
    
    def _get_network_io_metrics(self) -> Dict[str, Any]:
        """
        Get network I/O metrics (delta since last collection).
        
        Returns:
            Dictionary with network I/O metrics in MB
        """
        metrics = {
            'net_sent_mb': 0.0,
            'net_recv_mb': 0.0
        }
        
        try:
            current = psutil.net_io_counters()
            if current and self._last_net_io:
                # Calculate delta
                sent_bytes = current.bytes_sent - self._last_net_io.bytes_sent
                recv_bytes = current.bytes_recv - self._last_net_io.bytes_recv
                
                metrics['net_sent_mb'] = round(sent_bytes / (1024 * 1024), 2)
                metrics['net_recv_mb'] = round(recv_bytes / (1024 * 1024), 2)
                
            self._last_net_io = current
        except Exception as e:
            self.logger.warning(f"Error getting network I/O: {e}")
        
        return metrics
