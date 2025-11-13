"""
Base model interface for all ML models in the Crustacean Monitoring System.

This module provides an abstract base class that all model implementations
must extend. It implements the template method pattern to ensure consistent
model loading, inference, and cleanup across all models.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
from crustacean.utils.config import Config
from crustacean.utils.logging_setup import get_logger
from crustacean.utils.exceptions import ModelNotLoadedError, ModelLoadError, InferenceError


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    
    This class provides a consistent interface for model loading, inference,
    and resource management. All model implementations must extend this class
    and implement the abstract methods: load(), preprocess(), and postprocess().
    
    The predict() method implements the template method pattern, orchestrating
    the complete inference pipeline: preprocess → inference → postprocess.
    
    Attributes:
        config: Configuration object containing model settings
        interpreter: TFLite interpreter instance (None until loaded)
        input_details: Model input tensor details
        output_details: Model output tensor details
        logger: Logger instance for this model
        
    Example:
        >>> class MyModel(BaseModel):
        ...     def load(self):
        ...         # Load TFLite model
        ...     def preprocess(self, data):
        ...         # Preprocess input
        ...     def postprocess(self, output):
        ...         # Postprocess output
        ...
        >>> model = MyModel(config, preload=True)
        >>> result = model.predict(input_data)
        >>> model.unload()
        
        Or using context manager:
        >>> with MyModel(config) as model:
        ...     result = model.predict(input_data)
    """
    
    def __init__(self, config: Config, preload: bool = False):
        """
        Initialize the model with configuration.
        
        Args:
            config: Configuration object with model settings
            preload: If True, load model immediately. Default False (lazy loading)
            
        Example:
            >>> config = Config.load()
            >>> model = BinaryClassifier(config, preload=True)
        """
        self.config = config
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.logger = get_logger(self.__class__.__name__)
        
        if preload:
            self.load()
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the TFLite model and allocate tensors.
        
        This method must be implemented by subclasses to:
        1. Get model path from config
        2. Create TFLite interpreter
        3. Allocate tensors
        4. Store input and output details
        
        Raises:
            ModelLoadError: If model loading fails
            
        Example:
            >>> def load(self):
            ...     path = self.config.get('models.my_model.path')
            ...     self.interpreter = tflite.Interpreter(model_path=path)
            ...     self.interpreter.allocate_tensors()
            ...     self.input_details = self.interpreter.get_input_details()
            ...     self.output_details = self.interpreter.get_output_details()
        """
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> np.ndarray:
        """
        Preprocess input data for model inference.
        
        This method must be implemented by subclasses to transform
        raw input data into the format expected by the model.
        
        Args:
            input_data: Raw input data (frame, video, array, etc.)
            
        Returns:
            Preprocessed numpy array ready for inference
            
        Example:
            >>> def preprocess(self, frame):
            ...     # Resize and normalize
            ...     resized = cv2.resize(frame, (320, 180))
            ...     normalized = resized / 255.0
            ...     return np.expand_dims(normalized, axis=0)
        """
        pass
    
    @abstractmethod
    def postprocess(self, output_data: np.ndarray) -> Any:
        """
        Postprocess model output.
        
        This method must be implemented by subclasses to transform
        raw model output into the desired format.
        
        Args:
            output_data: Raw output from model inference
            
        Returns:
            Processed output in desired format
            
        Example:
            >>> def postprocess(self, output):
            ...     # Apply threshold and return binary result
            ...     return (output > 0.5).astype(int)
        """
        pass
    
    def predict(self, input_data: Any) -> Any:
        """
        Run inference on input data.
        
        This method implements the template method pattern, orchestrating
        the complete inference pipeline:
        1. Check model is loaded
        2. Preprocess input
        3. Run inference
        4. Postprocess output
        
        Args:
            input_data: Raw input data
            
        Returns:
            Processed model output
            
        Raises:
            ModelNotLoadedError: If model not loaded before calling predict
            InferenceError: If inference fails
            
        Example:
            >>> model = BinaryClassifier(config)
            >>> model.load()
            >>> result = model.predict(video_capture)
        """
        if self.interpreter is None:
            raise ModelNotLoadedError(
                f"{self.__class__.__name__} not loaded. Call load() first.",
                details={'model': self.__class__.__name__}
            )
        
        try:
            # Preprocess input
            preprocessed = self.preprocess(input_data)
            
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                preprocessed.astype(np.float32)
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            raw_output = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # Postprocess output
            return self.postprocess(raw_output)
            
        except Exception as e:
            raise InferenceError(
                f"Inference failed in {self.__class__.__name__}",
                details={'model': self.__class__.__name__, 'error': str(e)}
            ) from e
    
    def unload(self) -> None:
        """
        Release model resources.
        
        Clears the interpreter and tensor details, freeing memory.
        Safe to call multiple times.
        
        Example:
            >>> model.load()
            >>> # ... use model ...
            >>> model.unload()
        """
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.logger.info(f"{self.__class__.__name__} unloaded")
    
    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.
        
        Returns:
            True if model is loaded, False otherwise
            
        Example:
            >>> model = BinaryClassifier(config)
            >>> model.is_loaded()
            False
            >>> model.load()
            >>> model.is_loaded()
            True
        """
        return self.interpreter is not None
    
    def __enter__(self):
        """
        Context manager entry.
        
        Loads the model when entering the context.
        
        Returns:
            Self for use in with statement
            
        Example:
            >>> with BinaryClassifier(config) as model:
            ...     result = model.predict(data)
        """
        if not self.is_loaded():
            self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.
        
        Unloads the model when exiting the context, ensuring
        resources are always cleaned up.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
            
        Returns:
            False to propagate any exception
        """
        self.unload()
        return False
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "loaded" if self.is_loaded() else "not loaded"
        return f"{self.__class__.__name__}({status})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
