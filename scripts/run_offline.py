#!/usr/bin/env python3
"""
Offline Pipeline Entry Point Script.

This script provides a command-line interface for running the offline
(batch) video processing pipeline.

Usage:
    python scripts/run_offline.py --video-dir ./videos
    python scripts/run_offline.py --config config/custom.yaml --video-dir ./videos
    python scripts/run_offline.py --video-dir ./videos --profile --log-level DEBUG

Examples:
    # Process all videos in a directory
    python scripts/run_offline.py --video-dir ./processing/video

    # Use custom config and enable profiling
    python scripts/run_offline.py --config config/custom.yaml --video-dir ./videos --profile

    # Run with debug logging
    python scripts/run_offline.py --video-dir ./videos --log-level DEBUG
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from crustacean.utils.config import Config
from crustacean.utils.logging_setup import setup_logging, get_logger
from crustacean.utils.profiling import PerformanceProfiler
from crustacean.utils.exceptions import CrustaceanError
from crustacean.core.offline_pipeline import OfflinePipeline


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run the offline video processing pipeline",
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
        help="Output directory for results (overrides config)"
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
    
    # Setup logging
    setup_logging(config)
    logger = get_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("CRUSTACEAN OFFLINE PIPELINE")
    logger.info("=" * 50)
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Configuration: {args.config or 'default'}")
    logger.info(f"Profiling: {'enabled' if args.profile else 'disabled'}")
    
    # Validate video directory
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        logger.error(f"Video directory does not exist: {video_dir}")
        return 1
    
    if not video_dir.is_dir():
        logger.error(f"Path is not a directory: {video_dir}")
        return 1
    
    # Create profiler if enabled
    profiler = None
    if args.profile:
        profiler = PerformanceProfiler("offline_pipeline")
    
    # Run pipeline
    try:
        pipeline = OfflinePipeline(
            config=config,
            video_dir=str(video_dir),
            profiler=profiler
        )
        
        pipeline.run()
        
        logger.info("Pipeline completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except CrustaceanError as e:
        logger.error(f"Pipeline error: {e}")
        return 1
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
