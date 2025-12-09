"""
Base Pipeline class for the Crustacean Monitoring System.

This module provides an abstract base class that all pipeline implementations
must extend. It handles common functionality like model loading, cleanup,
and logging initialization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, TYPE_CHECKING

from crustacean.utils.config import Config
from crustacean.utils.logging_setup import get_logger
from crustacean.models.base_model import BaseModel

if TYPE_CHECKING:
    from crustacean.utils.profiling import PerformanceProfiler


class Pipeline(ABC):
    """
    Abstract base class for all pipeline modes.
    
    This class provides a consistent interface for pipeline implementations,
    handling common functionality like model management and logging.
    All pipeline implementations (Offline, Realtime) must extend this class
    and implement the abstract run() method.
    
    Attributes:
        config: Configuration object containing pipeline settings
        logger: Logger instance for this pipeline
        models: Dictionary of loaded model instances
        
    Example:
        >>> class MyPipeline(Pipeline):
        ...     def run(self):
        ...         self.load_models(preload=True)
        ...         try:
        ...             # Process data...
        ...         finally:
        ...             self.cleanup()
    """
    
    def __init__(self, config: Config, profiler: Optional['PerformanceProfiler'] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration object with pipeline settings
            profiler: Optional PerformanceProfiler for timing measurements
            
        Example:
            >>> config = Config.load()
            >>> profiler = PerformanceProfiler("offline")
            >>> pipeline = OfflinePipeline(config, video_dir='./videos', profiler=profiler)
        """
        self.config = config
        self.profiler = profiler
        self.logger = get_logger(self.__class__.__name__)
        self.models: Dict[str, BaseModel] = {}
        
        self.logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def run(self) -> None:
        """
        Execute the pipeline.
        
        This method must be implemented by subclasses to define
        the main processing logic for the pipeline.
        
        Raises:
            NotImplementedError: If not implemented by subclass
            
        Example:
            >>> pipeline = OfflinePipeline(config, video_dir='./videos')
            >>> pipeline.run()
        """
        pass
    
    def load_models(self, preload: bool = False) -> None:
        """
        Load all required models for the pipeline.
        
        Instantiates all four models (BinaryClassifier, FrameSelector,
        ObjectDetector, KeypointDetector) and optionally preloads them
        into memory.
        
        Args:
            preload: If True, load models into memory immediately.
                    If False, models will be loaded lazily on first use.
                    
        Example:
            >>> pipeline.load_models(preload=True)  # For real-time (keep in memory)
            >>> pipeline.load_models(preload=False)  # For offline (load per video)
        """
        # Import here to avoid circular imports
        from crustacean.models.binary_classifier import BinaryClassifier
        from crustacean.models.frame_selector import FrameSelector
        from crustacean.models.object_detector import ObjectDetector
        from crustacean.models.keypoint_detector import KeypointDetector
        
        self.logger.info(f"Loading models (preload={preload})")
        
        self.models['bc'] = BinaryClassifier(self.config, preload=preload)
        self.models['fs'] = FrameSelector(self.config, preload=preload)
        self.models['od'] = ObjectDetector(self.config, preload=preload)
        self.models['kd'] = KeypointDetector(self.config, preload=preload)
        
        loaded_count = sum(1 for m in self.models.values() if m.is_loaded())
        self.logger.info(f"Models instantiated: {len(self.models)}, preloaded: {loaded_count}")
    
    def cleanup(self) -> None:
        """
        Release all model resources.
        
        Unloads all models and clears the models dictionary.
        Safe to call multiple times.
        
        Example:
            >>> try:
            ...     pipeline.run()
            ... finally:
            ...     pipeline.cleanup()
        """
        self.logger.info("Cleaning up pipeline resources")
        
        for name, model in self.models.items():
            try:
                if model.is_loaded():
                    model.unload()
                    self.logger.debug(f"Unloaded model: {name}")
            except Exception as e:
                self.logger.warning(f"Error unloading model {name}: {e}")
        
        self.models.clear()
        self.logger.info("Pipeline cleanup complete")
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """
        Get a model by name.
        
        Args:
            name: Model identifier ('bc', 'fs', 'od', 'kd')
            
        Returns:
            Model instance or None if not found
            
        Example:
            >>> bc = pipeline.get_model('bc')
            >>> if bc and bc.is_loaded():
            ...     result = bc.predict(video)
        """
        return self.models.get(name)
    
    def is_models_loaded(self) -> bool:
        """
        Check if all models are loaded.
        
        Returns:
            True if all models are loaded, False otherwise
            
        Example:
            >>> if pipeline.is_models_loaded():
            ...     # Safe to run inference
        """
        if not self.models:
            return False
        return all(model.is_loaded() for model in self.models.values())
    
    def __enter__(self):
        """
        Context manager entry.
        
        Returns:
            Self for use in with statement
            
        Example:
            >>> with OfflinePipeline(config, video_dir) as pipeline:
            ...     pipeline.run()
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.
        
        Ensures cleanup is always called when exiting the context.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
            
        Returns:
            False to propagate any exception
        """
        self.cleanup()
        return False
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        model_status = "loaded" if self.is_models_loaded() else "not loaded"
        return f"{self.__class__.__name__}(models={model_status})"
