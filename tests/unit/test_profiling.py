"""
Unit tests for the performance profiling utilities.
"""

import pytest
import time
from crustacean.utils.profiling import PerformanceProfiler, TimingStats


class TestTimingStats:
    """Test TimingStats dataclass."""
    
    def test_initial_values(self):
        """Test initial values."""
        stats = TimingStats()
        assert stats.count == 0
        assert stats.total_time == 0.0
        assert stats.min_time == float('inf')
        assert stats.max_time == 0.0
        assert stats.times == []
    
    def test_add_timing(self):
        """Test adding timing measurements."""
        stats = TimingStats()
        stats.add_timing(1.0)
        stats.add_timing(2.0)
        stats.add_timing(3.0)
        
        assert stats.count == 3
        assert stats.total_time == 6.0
        assert stats.min_time == 1.0
        assert stats.max_time == 3.0
        assert stats.times == [1.0, 2.0, 3.0]
    
    def test_mean_time(self):
        """Test mean time calculation."""
        stats = TimingStats()
        stats.add_timing(1.0)
        stats.add_timing(2.0)
        stats.add_timing(3.0)
        
        assert stats.mean_time == 2.0
    
    def test_mean_time_empty(self):
        """Test mean time with no data."""
        stats = TimingStats()
        assert stats.mean_time == 0.0
    
    def test_std_dev(self):
        """Test standard deviation calculation."""
        stats = TimingStats()
        stats.add_timing(1.0)
        stats.add_timing(2.0)
        stats.add_timing(3.0)
        
        # std dev of [1, 2, 3] is 1.0
        assert abs(stats.std_dev - 1.0) < 0.01
    
    def test_std_dev_single_value(self):
        """Test std dev with single value."""
        stats = TimingStats()
        stats.add_timing(1.0)
        assert stats.std_dev == 0.0


class TestPerformanceProfiler:
    """Test PerformanceProfiler class."""
    
    def test_init(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler("test")
        
        assert profiler.name == "test"
        assert profiler.enabled is True
        assert profiler.sections == {}
    
    def test_init_disabled(self):
        """Test disabled profiler."""
        profiler = PerformanceProfiler("test", enabled=False)
        
        assert profiler.enabled is False
        assert profiler._start_time is None
    
    def test_profile_section(self):
        """Test profiling a section."""
        profiler = PerformanceProfiler("test")
        
        with profiler.profile_section("test_section"):
            time.sleep(0.01)
        
        assert "test_section" in profiler.sections
        stats = profiler.sections["test_section"]
        assert stats.count == 1
        assert stats.total_time >= 0.01
    
    def test_profile_section_multiple(self):
        """Test profiling same section multiple times."""
        profiler = PerformanceProfiler("test")
        
        for _ in range(3):
            with profiler.profile_section("test_section"):
                time.sleep(0.01)
        
        stats = profiler.sections["test_section"]
        assert stats.count == 3
        assert stats.total_time >= 0.03
    
    def test_profile_section_disabled(self):
        """Test profiling when disabled."""
        profiler = PerformanceProfiler("test", enabled=False)
        
        with profiler.profile_section("test_section"):
            time.sleep(0.01)
        
        # Section should not be recorded
        assert "test_section" not in profiler.sections
    
    def test_profile_multiple_sections(self):
        """Test profiling multiple different sections."""
        profiler = PerformanceProfiler("test")
        
        with profiler.profile_section("section_a"):
            time.sleep(0.01)
        
        with profiler.profile_section("section_b"):
            time.sleep(0.02)
        
        assert "section_a" in profiler.sections
        assert "section_b" in profiler.sections
        assert profiler.sections["section_b"].total_time > profiler.sections["section_a"].total_time
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        profiler = PerformanceProfiler("test")
        
        with profiler.profile_section("test_section"):
            time.sleep(0.01)
        
        summary = profiler.get_summary()
        
        assert "test_section" in summary
        assert summary["test_section"]["count"] == 1
        assert summary["test_section"]["total"] >= 0.01
        assert summary["test_section"]["mean"] >= 0.01
        assert summary["test_section"]["min"] >= 0.01
        assert summary["test_section"]["max"] >= 0.01
    
    def test_get_summary_empty(self):
        """Test getting summary with no data."""
        profiler = PerformanceProfiler("test")
        summary = profiler.get_summary()
        assert summary == {}
    
    def test_print_summary(self, capsys):
        """Test printing summary."""
        profiler = PerformanceProfiler("test")
        
        with profiler.profile_section("test_section"):
            time.sleep(0.01)
        
        output = profiler.print_summary()
        
        assert "PERFORMANCE PROFILE: test" in output
        assert "test_section" in output
        assert "Total runtime" in output
    
    def test_print_summary_empty(self, capsys):
        """Test printing summary with no data."""
        profiler = PerformanceProfiler("test")
        output = profiler.print_summary()
        
        assert "No sections recorded" in output
    
    def test_reset(self):
        """Test resetting profiler."""
        profiler = PerformanceProfiler("test")
        
        with profiler.profile_section("test_section"):
            time.sleep(0.01)
        
        assert len(profiler.sections) == 1
        
        profiler.reset()
        
        assert len(profiler.sections) == 0
    
    def test_get_section_stats(self):
        """Test getting stats for specific section."""
        profiler = PerformanceProfiler("test")
        
        with profiler.profile_section("test_section"):
            time.sleep(0.01)
        
        stats = profiler.get_section_stats("test_section")
        assert stats is not None
        assert stats.count == 1
        
        # Non-existent section
        assert profiler.get_section_stats("nonexistent") is None
    
    def test_repr(self):
        """Test string representation."""
        profiler = PerformanceProfiler("test")
        
        with profiler.profile_section("section"):
            pass
        
        repr_str = repr(profiler)
        assert "test" in repr_str
        assert "sections=1" in repr_str
        assert "enabled=True" in repr_str
    
    def test_exception_handling(self):
        """Test that timing works even with exceptions."""
        profiler = PerformanceProfiler("test")
        
        try:
            with profiler.profile_section("test_section"):
                time.sleep(0.01)
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Section should still be recorded
        assert "test_section" in profiler.sections
        assert profiler.sections["test_section"].count == 1


class TestProfilerIntegration:
    """Integration tests for profiler."""
    
    def test_nested_sections(self):
        """Test nested profiling sections."""
        profiler = PerformanceProfiler("test")
        
        with profiler.profile_section("outer"):
            time.sleep(0.01)
            with profiler.profile_section("inner"):
                time.sleep(0.01)
        
        assert profiler.sections["outer"].total_time >= 0.02
        assert profiler.sections["inner"].total_time >= 0.01
    
    def test_realistic_pipeline_profiling(self):
        """Test profiling a realistic pipeline scenario."""
        profiler = PerformanceProfiler("pipeline")
        
        # Simulate pipeline stages
        for i in range(3):
            with profiler.profile_section("preprocessing"):
                time.sleep(0.005)
            
            with profiler.profile_section("inference"):
                time.sleep(0.01)
            
            with profiler.profile_section("postprocessing"):
                time.sleep(0.003)
        
        summary = profiler.get_summary()
        
        # Verify all stages recorded
        assert summary["preprocessing"]["count"] == 3
        assert summary["inference"]["count"] == 3
        assert summary["postprocessing"]["count"] == 3
        
        # Verify inference is slowest
        assert summary["inference"]["total"] > summary["preprocessing"]["total"]
        assert summary["inference"]["total"] > summary["postprocessing"]["total"]
