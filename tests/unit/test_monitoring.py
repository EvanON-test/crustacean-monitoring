"""
Unit tests for the monitoring system.

Tests hardware detection, base monitor functionality, and
platform-specific monitor implementations.
"""

import pytest
import time
import csv
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import yaml

from crustacean.monitoring import (
    detect_hardware,
    create_monitor,
    BaseMonitor,
    JetsonMonitor,
    RaspberryPiMonitor,
    GenericMonitor
)
from crustacean.utils.config import Config


@pytest.fixture
def monitor_config_dict():
    """Configuration for monitoring tests."""
    return {
        'monitoring': {
            'interval': 1,
            'output_file': 'test_metrics.csv'
        },
        'logging': {
            'level': 'DEBUG',
            'console': True
        }
    }


@pytest.fixture
def monitor_config(temp_dir, monitor_config_dict):
    """Create Config object for monitoring tests."""
    config_path = temp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(monitor_config_dict, f)
    return Config.load(str(config_path))


class TestHardwareDetection:
    """Test hardware detection functionality."""
    
    def test_detect_hardware_returns_valid_type(self):
        """Test that detect_hardware returns a valid platform type."""
        result = detect_hardware()
        assert result in ['jetson', 'raspberry_pi', 'generic']
    
    @patch('platform.machine')
    @patch('os.path.exists')
    def test_detect_jetson(self, mock_exists, mock_machine):
        """Test Jetson detection."""
        mock_machine.return_value = 'aarch64'
        mock_exists.return_value = True  # /etc/nv_tegra_release exists
        
        result = detect_hardware()
        assert result == 'jetson'
    
    @patch('platform.machine')
    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    def test_detect_raspberry_pi(self, mock_open, mock_exists, mock_machine):
        """Test Raspberry Pi detection."""
        mock_machine.return_value = 'armv7l'
        
        def exists_side_effect(path):
            return path == '/proc/device-tree/model'
        
        mock_exists.side_effect = exists_side_effect
        mock_open.return_value.__enter__.return_value.read.return_value = 'Raspberry Pi 4'
        
        result = detect_hardware()
        assert result == 'raspberry_pi'
    
    @patch('platform.machine')
    @patch('os.path.exists')
    def test_detect_generic(self, mock_exists, mock_machine):
        """Test generic platform detection."""
        mock_machine.return_value = 'x86_64'
        mock_exists.return_value = False
        
        result = detect_hardware()
        assert result == 'generic'


class TestCreateMonitor:
    """Test monitor factory function."""
    
    @patch('crustacean.monitoring.hardware_detector.detect_hardware')
    def test_create_jetson_monitor(self, mock_detect, monitor_config, temp_dir):
        """Test creating JetsonMonitor."""
        mock_detect.return_value = 'jetson'
        
        output_file = str(temp_dir / 'metrics.csv')
        monitor = create_monitor(monitor_config, output_file)
        
        assert isinstance(monitor, JetsonMonitor)
    
    @patch('crustacean.monitoring.hardware_detector.detect_hardware')
    def test_create_pi_monitor(self, mock_detect, monitor_config, temp_dir):
        """Test creating RaspberryPiMonitor."""
        mock_detect.return_value = 'raspberry_pi'
        
        output_file = str(temp_dir / 'metrics.csv')
        monitor = create_monitor(monitor_config, output_file)
        
        assert isinstance(monitor, RaspberryPiMonitor)
    
    @patch('crustacean.monitoring.hardware_detector.detect_hardware')
    def test_create_generic_monitor(self, mock_detect, monitor_config, temp_dir):
        """Test creating GenericMonitor."""
        mock_detect.return_value = 'generic'
        
        output_file = str(temp_dir / 'metrics.csv')
        monitor = create_monitor(monitor_config, output_file)
        
        assert isinstance(monitor, GenericMonitor)


class TestBaseMonitor:
    """Test BaseMonitor functionality."""
    
    def test_get_common_metrics(self, monitor_config, temp_dir):
        """Test common metrics collection."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = GenericMonitor(monitor_config, output_file)
        
        metrics = monitor.get_common_metrics()
        
        assert 'timestamp' in metrics
        assert 'cpu_percent' in metrics
        assert 'ram_percent' in metrics
        assert 'ram_used_mb' in metrics
        
        # Verify types
        assert isinstance(metrics['cpu_percent'], (int, float))
        assert isinstance(metrics['ram_percent'], (int, float))
        assert isinstance(metrics['ram_used_mb'], (int, float))
    
    def test_get_fieldnames(self, monitor_config, temp_dir):
        """Test fieldnames include common fields."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = GenericMonitor(monitor_config, output_file)
        
        fieldnames = monitor.get_fieldnames()
        
        assert 'timestamp' in fieldnames
        assert 'cpu_percent' in fieldnames
        assert 'ram_percent' in fieldnames
        assert 'ram_used_mb' in fieldnames
    
    def test_stop_sets_event(self, monitor_config, temp_dir):
        """Test that stop() sets the stop event."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = GenericMonitor(monitor_config, output_file)
        
        assert not monitor.stop_event.is_set()
        monitor.stop()
        assert monitor.stop_event.is_set()


class TestGenericMonitor:
    """Test GenericMonitor functionality."""
    
    def test_collect_metrics(self, monitor_config, temp_dir):
        """Test metrics collection."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = GenericMonitor(monitor_config, output_file)
        
        metrics = monitor.collect_metrics()
        
        # Common metrics
        assert 'timestamp' in metrics
        assert 'cpu_percent' in metrics
        assert 'ram_percent' in metrics
        
        # Generic-specific metrics
        assert 'cpu_temp' in metrics
        assert 'disk_read_mb' in metrics
        assert 'disk_write_mb' in metrics
        assert 'net_sent_mb' in metrics
        assert 'net_recv_mb' in metrics
    
    def test_platform_fieldnames(self, monitor_config, temp_dir):
        """Test platform-specific fieldnames."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = GenericMonitor(monitor_config, output_file)
        
        fieldnames = monitor.get_platform_fieldnames()
        
        assert 'cpu_temp' in fieldnames
        assert 'disk_read_mb' in fieldnames
        assert 'disk_write_mb' in fieldnames
        assert 'net_sent_mb' in fieldnames
        assert 'net_recv_mb' in fieldnames
    
    def test_run_creates_csv(self, monitor_config, temp_dir):
        """Test that running monitor creates CSV file."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = GenericMonitor(monitor_config, output_file)
        monitor.interval = 0.1  # Fast interval for testing
        
        # Start monitor
        monitor.start()
        
        # Let it run briefly
        time.sleep(0.3)
        
        # Stop monitor
        monitor.stop()
        monitor.join(timeout=2)
        
        # Verify CSV was created
        assert Path(output_file).exists()
        
        # Verify CSV has content
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) >= 1
            
            # Verify header fields
            assert 'timestamp' in reader.fieldnames
            assert 'cpu_percent' in reader.fieldnames


class TestJetsonMonitor:
    """Test JetsonMonitor functionality."""
    
    def test_platform_fieldnames(self, monitor_config, temp_dir):
        """Test Jetson-specific fieldnames."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = JetsonMonitor(monitor_config, output_file)
        
        fieldnames = monitor.get_platform_fieldnames()
        
        assert 'cpu_temp' in fieldnames
        assert 'gpu_temp' in fieldnames
        assert 'gpu_percent' in fieldnames
        assert 'power_cur_mw' in fieldnames
        assert 'power_avg_mw' in fieldnames
        assert 'fan_speed' in fieldnames
    
    def test_collect_metrics_without_jtop(self, monitor_config, temp_dir):
        """Test metrics collection when jtop is unavailable."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = JetsonMonitor(monitor_config, output_file)
        
        # jtop_instance should be None on non-Jetson platforms
        metrics = monitor.collect_metrics()
        
        # Common metrics should still work
        assert 'timestamp' in metrics
        assert 'cpu_percent' in metrics
        
        # Jetson metrics should be N/A
        assert metrics['cpu_temp'] == 'N/A'
        assert metrics['gpu_temp'] == 'N/A'
    
    def test_empty_jetson_metrics(self, monitor_config, temp_dir):
        """Test empty Jetson metrics helper."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = JetsonMonitor(monitor_config, output_file)
        
        empty = monitor._get_empty_jetson_metrics()
        
        assert empty['cpu_temp'] == 'N/A'
        assert empty['gpu_temp'] == 'N/A'
        assert empty['gpu_percent'] == 'N/A'
        assert empty['power_cur_mw'] == 'N/A'
        assert empty['power_avg_mw'] == 'N/A'
        assert empty['fan_speed'] == 'N/A'


class TestRaspberryPiMonitor:
    """Test RaspberryPiMonitor functionality."""
    
    def test_platform_fieldnames(self, monitor_config, temp_dir):
        """Test Pi-specific fieldnames."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = RaspberryPiMonitor(monitor_config, output_file)
        
        fieldnames = monitor.get_platform_fieldnames()
        
        assert 'cpu_temp' in fieldnames
    
    def test_collect_metrics_without_gpiozero(self, monitor_config, temp_dir):
        """Test metrics collection when gpiozero is unavailable."""
        output_file = str(temp_dir / 'metrics.csv')
        monitor = RaspberryPiMonitor(monitor_config, output_file)
        
        metrics = monitor.collect_metrics()
        
        # Common metrics should still work
        assert 'timestamp' in metrics
        assert 'cpu_percent' in metrics
        
        # CPU temp should be N/A or a value from thermal zone
        assert 'cpu_temp' in metrics


class TestMonitorIntegration:
    """Integration tests for monitoring system."""
    
    def test_full_monitoring_cycle(self, monitor_config, temp_dir):
        """Test complete monitoring cycle."""
        output_file = str(temp_dir / 'metrics.csv')
        
        # Create monitor using factory
        monitor = create_monitor(monitor_config, output_file)
        monitor.interval = 0.1
        
        # Run monitoring
        monitor.start()
        time.sleep(0.5)
        monitor.stop()
        monitor.join(timeout=2)
        
        # Verify output
        assert Path(output_file).exists()
        
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Should have collected multiple samples
            assert len(rows) >= 2
            
            # Verify data integrity
            for row in rows:
                assert row['timestamp']
                assert float(row['cpu_percent']) >= 0
                assert float(row['ram_percent']) >= 0
