"""
NVIDIA Jetson hardware monitor.

This module provides hardware monitoring specific to NVIDIA Jetson
platforms (Nano, TX2, Xavier, Orin) using the jtop library.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING

from crustacean.monitoring.base_monitor import BaseMonitor

if TYPE_CHECKING:
    from crustacean.utils.config import Config

# Try to import jtop - may not be available on non-Jetson platforms
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False


class JetsonMonitor(BaseMonitor):
    """Hardware monitor for NVIDIA Jetson platforms."""
    
    def __init__(self, config: 'Config', output_file: str):
        """Initialize the Jetson monitor."""
        super().__init__(config, output_file)
        self.jtop_instance: Optional[Any] = None
        self._jtop_context = None
        
        if not JTOP_AVAILABLE:
            self.logger.warning(
                "jtop not available - Install with: pip install jetson-stats"
            )
    
    def run(self) -> None:
        """Main monitoring loop with jtop context management."""
        if JTOP_AVAILABLE:
            try:
                self._jtop_context = jtop()
                self.jtop_instance = self._jtop_context.__enter__()
                self.logger.info("jtop connection established")
            except Exception as e:
                self.logger.error(f"Failed to initialize jtop: {e}")
                self.jtop_instance = None
        
        try:
            super().run()
        finally:
            if self._jtop_context is not None:
                try:
                    self._jtop_context.__exit__(None, None, None)
                except Exception as e:
                    self.logger.error(f"Error closing jtop: {e}")

    def get_platform_fieldnames(self) -> List[str]:
        """Get Jetson-specific CSV field names."""
        return ['cpu_temp', 'gpu_temp', 'gpu_percent', 
                'power_cur_mw', 'power_avg_mw', 'fan_speed']
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect all hardware metrics including Jetson-specific ones."""
        metrics = self.get_common_metrics()
        
        if self.jtop_instance is not None:
            try:
                metrics.update(self._collect_jetson_metrics())
            except Exception as e:
                self.logger.warning(f"Error collecting Jetson metrics: {e}")
                metrics.update(self._get_empty_jetson_metrics())
        else:
            metrics.update(self._get_empty_jetson_metrics())
        
        return metrics
    
    def _collect_jetson_metrics(self) -> Dict[str, Any]:
        """Collect Jetson-specific metrics from jtop."""
        metrics = {}
        
        # CPU temperature
        try:
            stats = self.jtop_instance.stats
            if 'Temp CPU' in stats:
                metrics['cpu_temp'] = stats['Temp CPU']
            elif hasattr(self.jtop_instance, 'temperature'):
                temps = self.jtop_instance.temperature
                metrics['cpu_temp'] = temps.get('CPU', temps.get('cpu', 'N/A'))
            else:
                metrics['cpu_temp'] = 'N/A'
        except Exception:
            metrics['cpu_temp'] = 'N/A'
        
        # GPU temperature
        try:
            stats = self.jtop_instance.stats
            if 'Temp GPU' in stats:
                metrics['gpu_temp'] = stats['Temp GPU']
            elif hasattr(self.jtop_instance, 'temperature'):
                temps = self.jtop_instance.temperature
                metrics['gpu_temp'] = temps.get('GPU', temps.get('gpu', 'N/A'))
            else:
                metrics['gpu_temp'] = 'N/A'
        except Exception:
            metrics['gpu_temp'] = 'N/A'
        
        # GPU utilization
        try:
            if hasattr(self.jtop_instance, 'gpu'):
                gpu_info = self.jtop_instance.gpu
                if isinstance(gpu_info, dict):
                    metrics['gpu_percent'] = gpu_info.get('val', 'N/A')
                else:
                    metrics['gpu_percent'] = gpu_info
            else:
                metrics['gpu_percent'] = 'N/A'
        except Exception:
            metrics['gpu_percent'] = 'N/A'
        
        # Power consumption
        try:
            if hasattr(self.jtop_instance, 'power'):
                power = self.jtop_instance.power
                if isinstance(power, dict):
                    total = power.get('tot', power.get('total', {}))
                    if isinstance(total, dict):
                        metrics['power_cur_mw'] = total.get('cur', 'N/A')
                        metrics['power_avg_mw'] = total.get('avg', 'N/A')
                    else:
                        metrics['power_cur_mw'] = total
                        metrics['power_avg_mw'] = 'N/A'
                else:
                    metrics['power_cur_mw'] = 'N/A'
                    metrics['power_avg_mw'] = 'N/A'
            else:
                metrics['power_cur_mw'] = 'N/A'
                metrics['power_avg_mw'] = 'N/A'
        except Exception:
            metrics['power_cur_mw'] = 'N/A'
            metrics['power_avg_mw'] = 'N/A'
        
        # Fan speed
        try:
            if hasattr(self.jtop_instance, 'fan'):
                fan = self.jtop_instance.fan
                if isinstance(fan, dict):
                    for key, value in fan.items():
                        if isinstance(value, dict):
                            metrics['fan_speed'] = value.get('speed', 'N/A')
                        else:
                            metrics['fan_speed'] = value
                        break
                else:
                    metrics['fan_speed'] = fan
            else:
                metrics['fan_speed'] = 'N/A'
        except Exception:
            metrics['fan_speed'] = 'N/A'
        
        return metrics
    
    def _get_empty_jetson_metrics(self) -> Dict[str, Any]:
        """Get empty Jetson metrics when jtop is unavailable."""
        return {
            'cpu_temp': 'N/A',
            'gpu_temp': 'N/A',
            'gpu_percent': 'N/A',
            'power_cur_mw': 'N/A',
            'power_avg_mw': 'N/A',
            'fan_speed': 'N/A'
        }
