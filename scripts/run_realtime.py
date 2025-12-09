#!/usr/bin/env python3
"""
Real-time Pipeline Entry Point Script.

This script provides a command-line interface for running the real-time
camera processing pipeline.

Usage:
    python scripts/run_realtime.py
    python scripts/run_realtime.py --display
    python scripts/run_realtime.py --config config/custom.yaml --profile

Examples:
    # Run in headless mode (no display)
    python scripts/run_realtime.py

    # Run with video display
    python scripts/run_realtime.py --display

    # Use custom config and enable profiling
    python scripts/run_realtime.py --config config/custom.yaml --profile

    # Run with debug logging
    python scripts/run_realtime.py --log-level DEBUG
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from crustacean.utils.config import Config
from crustacean.utils.logging_setup import setup_logging, get_logger
from crustacean.utils.profiling import PerformanceProfiler
from crustacean.utils.exceptions import CrustaceanError, CameraInitError
from crustacean.core.realtime_pipeline import RealtimePipeline


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run the real-time camera processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file (default: config/default_config.yaml)"
    )
    
    parser.add_argument(
        "--display", "-d",
        action="store_true",
        help="Enable video display with overlays"
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
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for detections (overrides config)"
    )
    
    parser.add_argument(
        "--camera-type",
        type=str,
        choices=["csi", "usb"],
        default=None,
        help="Camera type (overrides config)"
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
    
    # Override output directory if specified
    if args.output_dir:
        config._config.setdefault('output', {})['detections_dir'] = args.output_dir
    
    # Override camera type if specified
    if args.camera_type:
        config._config.setdefault('camera', {})['type'] = args.camera_type
    
    # Setup logging
    setup_logging(config)
    logger = get_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("CRUSTACEAN REAL-TIME PIPELINE")
    logger.info("=" * 50)
    logger.info(f"Display mode: {'enabled' if args.display else 'disabled'}")
    logger.info(f"Configuration: {args.config or 'default'}")
    logger.info(f"Profiling: {'enabled' if args.profile else 'disabled'}")
    logger.info("Press Ctrl+C to stop")
    
    # Create profiler if enabled
    profiler = None
    if args.profile:
        profiler = PerformanceProfiler("realtime_pipeline")
    
    # Run pipeline
    try:
        pipeline = RealtimePipeline(
            config=config,
            display_mode=args.display,
            profiler=profiler
        )
        
        pipeline.run()
        
        logger.info("Pipeline completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except CameraInitError as e:
        logger.error(f"Camera initialization failed: {e}")
        logger.error("Check camera connection and configuration")
        return 2
        
    except CrustaceanError as e:
        logger.error(f"Pipeline error: {e}")
        return 1
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
