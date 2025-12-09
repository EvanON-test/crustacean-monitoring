"""
Keypoint Detector model for detecting anatomical keypoints in crustacean ROIs.

This module implements the KeypointDetector model which detects anatomical
keypoints (7 points with x,y coordinates) in cropped ROI regions from the
object detector.
"""

from typing import Any
import numpy as np
import tflite_runtime.interpreter as tflite

from crustacean.models.base_model import BaseModel
from crustacean.utils.exceptions import ModelLoadError, InferenceError


class KeypointDetector(BaseModel):
    """
    Detects anatomical keypoints in cropped ROI regions.
    
    This model processes cropped ROI frames from the object detector
    and returns keypoint coordinates for anatomical landmarks on
    crustaceans. It supports both single frame and batch processing.
    
    The model expects:
    - Grayscale ROI frames of shape (height, width) or batch (n, height, width)
    
    Returns:
    - Array of keypoint coordinates with shape (n_frames, 14)
      where 14 = 7 keypoints × 2 coordinates (x, y)
    
    Attributes:
        num_keypoints: Number of keypoints to detect (default 7)
        
    Example:
        >>> config = Config.load()
        >>> kd = KeypointDetector(config, preload=True)
        >>> # Single frame
        >>> keypoints = kd.predict(roi_frame)
        >>> # Batch processing
        >>> keypoints = kd.predict(roi_frames)  # shape (n, h, w)
        >>> print(keypoints.shape)  # (n, 14)
    """
    
    def __init__(self, config, preload: bool = False):
        """
        Initialize KeypointDetector with configuration.
        
        Args:
            config: Configuration object with model settings
            preload: If True, load model immediately
        """
        # Initialize base class without preloading
        super().__init__(config, preload=False)
        
        # Get configuration
        self.num_keypoints = self.config.get(
            'models.keypoint_detector.num_keypoints', 7
        )
        
        if preload:
            self.load()
    
    def load(self) -> None:
        """
        Load the TFLite keypoint detection model.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            model_path = self.config.get('models.keypoint_detector.path')
            if not model_path:
                raise ModelLoadError(
                    "Keypoint detector model path not found in configuration",
                    details={'config_key': 'models.keypoint_detector.path'}
                )
            
            self.logger.info(f"Loading keypoint detector from {model_path}")
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.logger.info("KeypointDetector model loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load KeypointDetector model: {str(e)}",
                details={'error': str(e)}
            ) from e
    
    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        """
        Preprocess ROI frames for keypoint detection.
        
        Handles both single frames and batches. Reshapes frames
        to the format expected by the model.
        
        Args:
            frames: Grayscale ROI frame(s)
                - Single frame: shape (height, width)
                - Batch: shape (n_frames, height, width)
                
        Returns:
            Preprocessed frames ready for inference
            Shape: (1, height, width, 1) for single frame processing
        """
        # Ensure frames is at least 3D (batch dimension)
        if frames.ndim == 2:
            # Single frame: (h, w) -> (1, h, w)
            frames = np.expand_dims(frames, axis=0)
        
        return frames
    
    def postprocess(self, output_data: np.ndarray) -> np.ndarray:
        """
        Postprocess model output to get keypoint coordinates.
        
        Args:
            output_data: Raw model output
            
        Returns:
            Keypoint coordinates array
        """
        # Output is already in the correct format (14 values per frame)
        return output_data.flatten()
    
    def predict(self, frames: np.ndarray) -> np.ndarray:
        """
        Detect keypoints in ROI frame(s).
        
        Supports both single frame and batch processing. For batch
        processing, each frame is processed sequentially through
        the model.
        
        Args:
            frames: Grayscale ROI frame(s)
                - Single frame: shape (height, width)
                - Batch: shape (n_frames, height, width)
                
        Returns:
            Keypoint coordinates array
            - Single frame: shape (14,) - 7 keypoints × 2 coords
            - Batch: shape (n_frames, 14)
            
        Raises:
            InferenceError: If keypoint detection fails
            
        Example:
            >>> # Single frame
            >>> kp = kd.predict(roi)
            >>> print(kp.shape)  # (14,)
            >>> 
            >>> # Batch of 5 frames
            >>> kps = kd.predict(rois)  # rois.shape = (5, 539, 561)
            >>> print(kps.shape)  # (5, 14)
        """
        if self.interpreter is None:
            raise InferenceError(
                "KeypointDetector model not loaded",
                details={'model': 'KeypointDetector'}
            )
        
        try:
            # Preprocess to ensure batch dimension
            frames = self.preprocess(frames)
            n_frames = frames.shape[0]
            
            # Initialize output array
            coords = np.zeros((n_frames, self.num_keypoints * 2))
            
            self.logger.debug(f"Processing {n_frames} frames for keypoint detection")
            
            # Process each frame
            for i in range(n_frames):
                frame = frames[i]
                
                # Reshape for model input: (1, height, width, 1)
                input_data = np.reshape(
                    frame, 
                    (1, frame.shape[0], frame.shape[1], 1)
                )
                
                # Set input tensor
                self.interpreter.set_tensor(
                    self.input_details[0]['index'],
                    input_data.astype(np.float32)
                )
                
                # Run inference
                self.interpreter.invoke()
                
                # Get output
                output = self.interpreter.get_tensor(
                    self.output_details[0]['index']
                )
                
                # Store coordinates
                coords[i] = output.flatten()
            
            self.logger.debug(f"Keypoint detection complete: {coords.shape}")
            
            # Return single array if single frame input
            if n_frames == 1:
                return coords[0]
            
            return coords
            
        except Exception as e:
            raise InferenceError(
                f"Keypoint detection failed: {str(e)}",
                details={'model': 'KeypointDetector', 'error': str(e)}
            ) from e
    
    def get_keypoint_pairs(self, coords: np.ndarray) -> np.ndarray:
        """
        Reshape flat coordinates into (x, y) pairs.
        
        Utility method to convert the flat coordinate array
        into a more usable format of (x, y) pairs.
        
        Args:
            coords: Flat coordinate array of shape (14,) or (n, 14)
            
        Returns:
            Reshaped array of shape (7, 2) or (n, 7, 2)
            
        Example:
            >>> kp = kd.predict(roi)
            >>> pairs = kd.get_keypoint_pairs(kp)
            >>> print(pairs.shape)  # (7, 2)
            >>> print(pairs[0])  # [x0, y0] - first keypoint
        """
        if coords.ndim == 1:
            return coords.reshape(self.num_keypoints, 2)
        else:
            return coords.reshape(-1, self.num_keypoints, 2)
