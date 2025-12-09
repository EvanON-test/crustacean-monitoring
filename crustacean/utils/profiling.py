"""
Performance profiling utilities.

This module provides tools for measuring and reporting execution times
of different pipeline stages and operations.
"""

import time
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Generator

from crustacean.utils.logging_setup import get_logger


@dataclass
class TimingStats:
    """Statistics for a profiled section."""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)
    
    @property
    def mean_time(self) -> float:
        """Calculate mean execution time."""
        if self.count == 0:
            return 0.0
        return self.total_time / self.count
    
    @property
    def std_dev(self) -> float:
        """Calculate standard deviation of execution times."""
        if len(self.times) < 2:
            return 0.0
        return statistics.stdev(self.times)
    
    def add_timing(self, elapsed: float) -> None:
        """Add a timing measurement."""
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)


class PerformanceProfiler:
    """
    Performance profiler for measuring execution times.
    
    Provides context managers for timing code sections and methods
    for generating summary statistics and reports.
    
    Attributes:
        name: Name of this profiler instance
        sections: Dictionary of section names to timing statistics
        enabled: Whether profiling is active
        
    Example:
        >>> profiler = PerformanceProfiler("pipeline")
        >>> 
        >>> with profiler.profile_section("preprocessing"):
        ...     preprocess_data()
        >>> 
        >>> with profiler.profile_section("inference"):
        ...     run_model()
        >>> 
        >>> profiler.print_summary()
    """
    
    def __init__(self, name: str = "profiler", enabled: bool = True):
        """
        Initialize the profiler.
        
        Args:
            name: Name for this profiler instance
            enabled: If False, profiling is disabled (no-op)
        """
        self.name = name
        self.enabled = enabled
        self.sections: Dict[str, TimingStats] = {}
        self.logger = get_logger(self.__class__.__name__)
        self._start_time: Optional[float] = None
        
        if enabled:
            self._start_time = time.time()
            self.logger.info(f"Profiler '{name}' initialized")
    
    @contextmanager
    def profile_section(self, section_name: str) -> Generator[None, None, None]:
        """
        Context manager for timing a code section.
        
        Args:
            section_name: Name to identify this section in reports
            
        Yields:
            None
            
        Example:
            >>> with profiler.profile_section("model_inference"):
            ...     result = model.predict(data)
        """
        if not self.enabled:
            yield
            return
        
        # Initialize section if needed
        if section_name not in self.sections:
            self.sections[section_name] = TimingStats()
        
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.sections[section_name].add_timing(elapsed)
            
            self.logger.debug(
                f"[{section_name}] completed in {elapsed*1000:.2f}ms"
            )
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all profiled sections.
        
        Returns:
            Dictionary mapping section names to their statistics:
            - count: Number of times section was executed
            - total: Total time in seconds
            - mean: Mean time in seconds
            - min: Minimum time in seconds
            - max: Maximum time in seconds
            - std: Standard deviation in seconds
        """
        summary = {}
        
        for name, stats in self.sections.items():
            summary[name] = {
                'count': stats.count,
                'total': stats.total_time,
                'mean': stats.mean_time,
                'min': stats.min_time if stats.count > 0 else 0.0,
                'max': stats.max_time,
                'std': stats.std_dev
            }
        
        return summary
    
    def print_summary(self, include_header: bool = True) -> str:
        """
        Print formatted summary of profiling results.
        
        Args:
            include_header: Whether to include header line
            
        Returns:
            Formatted summary string
        """
        if not self.sections:
            msg = f"Profiler '{self.name}': No sections recorded"
            print(msg)
            return msg
        
        lines = []
        
        if include_header:
            total_runtime = 0.0
            if self._start_time:
                total_runtime = time.time() - self._start_time
            
            lines.append("=" * 70)
            lines.append(f"PERFORMANCE PROFILE: {self.name}")
            lines.append(f"Total runtime: {total_runtime:.2f}s")
            lines.append("=" * 70)
        
        # Header row
        lines.append(
            f"{'Section':<25} {'Count':>8} {'Total':>10} {'Mean':>10} "
            f"{'Min':>10} {'Max':>10}"
        )
        lines.append("-" * 70)
        
        # Sort sections by total time (descending)
        sorted_sections = sorted(
            self.sections.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )
        
        for name, stats in sorted_sections:
            lines.append(
                f"{name:<25} {stats.count:>8} "
                f"{stats.total_time:>9.3f}s "
                f"{stats.mean_time*1000:>9.2f}ms "
                f"{stats.min_time*1000:>9.2f}ms "
                f"{stats.max_time*1000:>9.2f}ms"
            )
        
        lines.append("=" * 70)
        
        output = "\n".join(lines)
        print(output)
        self.logger.info(f"Profile summary:\n{output}")
        
        return output
    
    def reset(self) -> None:
        """Reset all timing data."""
        self.sections.clear()
        self._start_time = time.time() if self.enabled else None
        self.logger.debug(f"Profiler '{self.name}' reset")
    
    def get_section_stats(self, section_name: str) -> Optional[TimingStats]:
        """
        Get statistics for a specific section.
        
        Args:
            section_name: Name of the section
            
        Returns:
            TimingStats for the section, or None if not found
        """
        return self.sections.get(section_name)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PerformanceProfiler(name='{self.name}', "
            f"sections={len(self.sections)}, enabled={self.enabled})"
        )
