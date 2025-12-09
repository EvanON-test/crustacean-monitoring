#!/usr/bin/env python3
"""
Monitoring Pipeline Entry Point Script.

This script runs the offline pipeline with hardware monitoring enabled,
collecting system metrics (CPU, RAM, temperature) during processing.

Usage:
    python scripts/run_monitoring.py --video-dir ./videos
    python scripts/run_monitoring.py --video-dir ./videos --output metrics.csv
    python scripts/run_monitoring.py --config config/custom.yaml --video-dir ./videos

Examples:
    # Process videos with default monitoring output
    python scripts/run_monitoring.py --video-dir ./processing/video

    # Specify custom metrics output file
    python scripts/run_monitoring.py --video-dir ./videos --output benchmark/metrics.csv

    # Use custom config
    python scripts/run_monitoring.py --config config/custom.yaml --video-dir ./videos
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from crustacean.utils.config import Config
from crustacean.utils.logging_setup import setup_logging, get_logger
from crustacean.utils.profiling import PerformanceProfiler
from crustacean.utils.exceptions import CrustaceanError
from crustacean.core.offline_pipeline import OfflinePipeline
from crustacean.monitoring import create_monitor, detect_hardware


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run the offline pipeline with hardware monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--video-dir", "-v",
        type=str,
        required=True,
        help="Directory containing video files to process"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file (default: config/default_config.yaml)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="monitoring_metrics.csv",
        help="Output file for monitoring metrics (default: monitoring_metrics.csv)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--profile", "-p",
        action="store_true",
        help="Enable performance profiling"
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Monitoring interval in seconds (default: 2.0)"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    args = parse_args()
    
    # Load configuration
    try:
        config = Config.load(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    # Override log level if specified
    if args.log_level:
        config._config.setdefault('logging', {})['level'] = args.log_level
    
    # Set monitoring interval
    config._config.setdefault('monitoring', {})['interval'] = args.interval
    
    # Setup logging
    setup_logging(config)
    logger = get_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("CRUSTACEAN MONITORING PIPELINE")
    logger.info("=" * 50)
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Configuration: {args.config or 'default'}")
    logger.info(f"Metrics output: {args.output}")
    logger.info(f"Monitoring interval: {args.interval}s")
    logger.info(f"Profiling: {'enabled' if args.profile else 'disabled'}")
    
    # Detect hardware platform
    hardware = detect_hardware()
    logger.info(f"Detected hardware: {hardware}")
    
    # Validate video directory
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        logger.error(f"Video directory does not exist: {video_dir}")
        return 1
    
    if not video_dir.is_dir():
        logger.error(f"Path is not a directory: {video_dir}")
        return 1
    
    # Create output directory for metrics if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create profiler if enabled
    profiler = None
    if args.profile:
        profiler = PerformanceProfiler("monitoring_pipeline")
    
    # Create monitor
    monitor = create_monitor(config, str(output_path))
    
    # Run pipeline with monitoring
    try:
        # Start monitoring thread
        logger.info("Starting hardware monitor")
        monitor.start()
        
        # Create and run pipeline
        pipeline = OfflinePipeline(
            config=config,
            video_dir=str(video_dir),
            profiler=profiler
        )
        
        start_time = time.time()
        pipeline.run()
        elapsed = time.time() - start_time
        
        logger.info(f"Pipeline completed in {elapsed:.2f}s")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
        
    except CrustaceanError as e:
        logger.error(f"Pipeline error: {e}")
        return 1
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1
        
    finally:
        # Stop monitoring
        logger.info("Stopping hardware monitor")
        monitor.stop()
        monitor.join(timeout=5)
        
        if monitor.is_alive():
            logger.warning("Monitor thread did not stop gracefully")
        else:
            logger.info(f"Monitoring metrics saved to: {args.output}")


if __name__ == "__main__":
    sys.exit(main())
