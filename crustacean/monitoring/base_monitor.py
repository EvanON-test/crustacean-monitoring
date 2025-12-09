"""
Base hardware monitoring class.

This module provides the abstract base class for all hardware monitors,
implementing common functionality for metric collection and CSV output.
"""

import csv
import time
from abc import ABC, abstractmethod
from threading import Thread, Event
from typing import Dict, Any, List, TYPE_CHECKING

import psutil

from crustacean.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from crustacean.utils.config import Config


class BaseMonitor(Thread, ABC):
    """
    Abstract base class for hardware monitoring.
    
    Runs as a background thread, periodically collecting hardware metrics
    and writing them to a CSV file. Subclasses implement platform-specific
    metric collection.
    
    Attributes:
        config: Configuration object
        output_file: Path to output CSV file
        interval: Seconds between metric collections
        stop_event: Event to signal thread shutdown
        
    Example:
        >>> monitor = JetsonMonitor(config, 'metrics.csv')
        >>> monitor.start()
        >>> # ... run pipeline ...
        >>> monitor.stop()
        >>> monitor.join()
    """
    
    def __init__(self, config: 'Config', output_file: str):
        """
        Initialize the monitor.
        
        Args:
            config: Configuration object with monitoring settings
            output_file: Path to output CSV file for metrics
        """
        super().__init__(name="MonitorThread", daemon=True)
        
        self.config = config
        self.output_file = output_file
        self.interval = config.get('monitoring.interval', 2)
        self.stop_event = Event()
        self.logger = get_logger(self.__class__.__name__)
        
        # Track metrics for summary
        self._metrics_count = 0
        self._start_time = None
        
        self.logger.info(
            f"Monitor initialized: output={output_file}, interval={self.interval}s"
        )
    
    def run(self) -> None:
        """
        Main monitoring loop.
        
        Collects metrics at regular intervals and writes to CSV file.
        Continues until stop() is called.
        """
        self._start_time = time.time()
        self.logger.info("Monitor thread started")
        
        try:
            with open(self.output_file, 'w', newline='') as csvfile:
                fieldnames = self.get_fieldnames()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                csvfile.flush()
                
                while not self.stop_event.wait(self.interval):
                    try:
                        metrics = self.collect_metrics()
                        writer.writerow(metrics)
                        csvfile.flush()
                        self._metrics_count += 1
                        
                        self.logger.debug(
                            f"Collected metrics: CPU={metrics.get('cpu_percent', 'N/A')}%, "
                            f"RAM={metrics.get('ram_percent', 'N/A')}%"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Error collecting metrics: {e}")
                        
        except IOError as e:
            self.logger.error(f"Failed to write metrics file: {e}")
            
        except Exception as e:
            self.logger.exception(f"Monitor thread error: {e}")
            
        finally:
            self._log_summary()
            self.logger.info("Monitor thread stopped")
    
    def stop(self) -> None:
        """
        Signal the monitor thread to stop.
        
        The thread will complete its current collection cycle and then exit.
        Call join() after stop() to wait for the thread to finish.
        """
        self.logger.info("Stop requested")
        self.stop_event.set()
    
    def get_fieldnames(self) -> List[str]:
        """
        Get CSV column names for metrics.
        
        Returns:
            List of field names for CSV header
        """
        # Common fields + platform-specific fields
        common = ['timestamp', 'cpu_percent', 'ram_percent', 'ram_used_mb']
        platform_specific = self.get_platform_fieldnames()
        return common + platform_specific
    
    def get_platform_fieldnames(self) -> List[str]:
        """
        Get platform-specific CSV field names.
        
        Override in subclasses to add platform-specific metrics.
        
        Returns:
            List of platform-specific field names
        """
        return []
    
    def get_common_metrics(self) -> Dict[str, Any]:
        """
        Collect metrics available on all platforms.
        
        Returns:
            Dictionary with common hardware metrics
        """
        memory = psutil.virtual_memory()
        
        return {
            'timestamp': time.strftime("%Y-%m-%d_%H-%M-%S"),
            'cpu_percent': psutil.cpu_percent(interval=None),
            'ram_percent': memory.percent,
            'ram_used_mb': memory.used / (1024 * 1024)
        }
    
    @abstractmethod
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect all hardware metrics.
        
        Must be implemented by subclasses to collect platform-specific
        metrics in addition to common metrics.
        
        Returns:
            Dictionary with all collected metrics
        """
        pass
    
    def _log_summary(self) -> None:
        """Log monitoring session summary."""
        if self._start_time:
            runtime = time.time() - self._start_time
            self.logger.info(
                f"Monitoring summary: {self._metrics_count} samples "
                f"collected over {runtime:.1f}s"
            )
