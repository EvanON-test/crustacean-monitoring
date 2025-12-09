"""
Frame Selector model for identifying highest quality frames in video segments.

This module implements the FrameSelector model which analyzes video frames
to select the best quality frames from continuous segments where crustaceans
are detected. It uses two separate models (top and bottom) to assess frame quality.
"""

from typing import List, Tuple, Any
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

from crustacean.models.base_model import BaseModel
from crustacean.utils.exceptions import ModelLoadError, InferenceError


class FrameSelector(BaseModel):
    """
    Selects highest quality frames from video segments.
    
    This model processes video frames to identify the best quality frames
    within continuous segments where crustaceans are detected. It uses two
    separate TFLite models (top and bottom) to assess frame quality and
    returns the indices of the best frames for each segment.
    
    The model expects:
    - A binary signal indicating crustacean presence per frame
    - A video capture object to read frames from
    
    Returns:
    - Lists of frame indices for top and bottom model selections
    
    Attributes:
        top_interpreter: TFLite interpreter for top quality model
        bottom_interpreter: TFLite interpreter for bottom quality model
        top_input_details: Input tensor details for top model
        top_output_details: Output tensor details for top model
        bottom_input_details: Input tensor details for bottom model
        bottom_output_details: Output tensor details for bottom model
        input_width: Width for frame rescaling
        input_height: Height for frame rescaling
        
    Example:
        >>> config = Config.load()
        >>> fs = FrameSelector(config, preload=True)
        >>> signal = np.array([0, 1, 1, 1, 0, 1, 1, 0])
        >>> best_frames = fs.predict(signal, video_capture)
        >>> print(best_frames)  # [[2], [5]]  # top and bottom selections
    """
    
    def __init__(self, config, preload: bool = False):
        """
        Initialize FrameSelector with configuration.
        
        Args:
            config: Configuration object with model settings
            preload: If True, load models immediately
        """
        # Initialize base class without preloading
        super().__init__(config, preload=False)
        
        # FrameSelector-specific attributes
        self.top_interpreter = None
        self.bottom_interpreter = None
        self.top_input_details = None
        self.top_output_details = None
        self.bottom_input_details = None
        self.bottom_output_details = None
        
        # Get configuration
        self.input_width = self.config.get('models.frame_selector.input_width', 320)
        self.input_height = self.config.get('models.frame_selector.input_height', 180)
        
        if preload:
            self.load()
    
    def load(self) -> None:
        """
        Load both top and bottom TFLite models.
        
        Raises:
            ModelLoadError: If either model fails to load
        """
        try:
            # Load top quality model
            top_model_path = self.config.get('models.frame_selector.top_model_path')
            if not top_model_path:
                raise ModelLoadError(
                    "Top model path not found in configuration",
                    details={'config_key': 'models.frame_selector.top_model_path'}
                )
            
            self.logger.info(f"Loading top model from {top_model_path}")
            self.top_interpreter = tflite.Interpreter(model_path=top_model_path)
            self.top_interpreter.allocate_tensors()
            self.top_input_details = self.top_interpreter.get_input_details()
            self.top_output_details = self.top_interpreter.get_output_details()
            
            # Load bottom quality model
            bottom_model_path = self.config.get('models.frame_selector.bottom_model_path')
            if not bottom_model_path:
                raise ModelLoadError(
                    "Bottom model path not found in configuration",
                    details={'config_key': 'models.frame_selector.bottom_model_path'}
                )
            
            self.logger.info(f"Loading bottom model from {bottom_model_path}")
            self.bottom_interpreter = tflite.Interpreter(model_path=bottom_model_path)
            self.bottom_interpreter.allocate_tensors()
            self.bottom_input_details = self.bottom_interpreter.get_input_details()
            self.bottom_output_details = self.bottom_interpreter.get_output_details()
            
            # Set base class interpreter to top (for is_loaded check)
            self.interpreter = self.top_interpreter
            self.input_details = self.top_input_details
            self.output_details = self.top_output_details
            
            self.logger.info("FrameSelector models loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load FrameSelector models: {str(e)}",
                details={'error': str(e)}
            ) from e
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame for quality assessment.
        
        Converts frame to grayscale, rescales to model input size,
        and reshapes for TFLite inference.
        
        Args:
            frame: BGR frame from video (H, W, 3)
            
        Returns:
            Preprocessed frame ready for inference (1, H, W, 1)
        """
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Rescale to model input size
        rescaled = cv2.resize(gray_frame, (self.input_width, self.input_height))
        
        # Reshape for TFLite: (1, height, width, 1)
        reshaped = np.reshape(rescaled, (1, rescaled.shape[0], rescaled.shape[1], 1))
        
        return reshaped
    
    def postprocess(self, output_data: np.ndarray) -> float:
        """
        Postprocess model output to get quality score.
        
        Args:
            output_data: Raw model output
            
        Returns:
            Quality score as float
        """
        # Return the quality prediction value
        return float(output_data[0][0])
    
    def _predict_quality(self, interpreter, input_details, output_details, frame: np.ndarray) -> float:
        """
        Predict quality score for a single frame using specified model.
        
        Args:
            interpreter: TFLite interpreter to use
            input_details: Input tensor details
            output_details: Output tensor details
            frame: Preprocessed frame
            
        Returns:
            Quality score
        """
        # Set input tensor
        interpreter.set_tensor(
            input_details[0]['index'],
            frame.astype(np.float32)
        )
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        return float(prediction[0][0])
    
    def predict(self, signal: np.ndarray, video: cv2.VideoCapture) -> List[List[int]]:
        """
        Select best frames from video based on binary signal.
        
        This method identifies continuous segments where the signal indicates
        crustacean presence (value = 1), evaluates frame quality within each
        segment using both top and bottom models, and returns the indices of
        the best frames.
        
        Args:
            signal: Binary array indicating crustacean presence per frame (0 or 1)
            video: OpenCV VideoCapture object positioned at start
            
        Returns:
            List of two lists: [top_frame_indices, bottom_frame_indices]
            Each inner list contains frame indices of best frames per segment
            
        Raises:
            InferenceError: If frame reading or quality prediction fails
            
        Example:
            >>> signal = np.array([0, 1, 1, 1, 0, 1, 1, 0])
            >>> best_frames = fs.predict(signal, video)
            >>> print(best_frames)
            [[2, 5], [2, 6]]  # Frame 2 best in segment 1, frame 5 best in segment 2 (top)
                               # Frame 2 best in segment 1, frame 6 best in segment 2 (bottom)
        """
        if self.top_interpreter is None or self.bottom_interpreter is None:
            raise InferenceError(
                "FrameSelector models not loaded",
                details={'model': 'FrameSelector'}
            )
        
        try:
            # Read first frame to verify video is readable
            success, image = video.read()
            if not success:
                raise InferenceError(
                    "Failed to read first frame from video",
                    details={'model': 'FrameSelector'}
                )
            
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Arrays for quality scores within current segment
            contig_t = []  # Top model scores
            contig_b = []  # Bottom model scores
            in_contig = False
            
            # Index of best frames for current segment
            current_best_t = None
            current_best_b = None
            
            # Array of best frame indices (what gets returned)
            best_frames = [[], []]  # [top_indices, bottom_indices]
            
            self.logger.debug(f"Processing {total_frames} frames for quality assessment")
            
            # Loop over all frames
            for i in range(total_frames):
                # If crustacean present (signal = 1)
                if signal[i]:
                    in_contig = True
                    
                    # Preprocess frame
                    preprocessed_frame = self.preprocess(image)
                    
                    # Assess quality using both models
                    t_quality = self._predict_quality(
                        self.top_interpreter,
                        self.top_input_details,
                        self.top_output_details,
                        preprocessed_frame
                    )
                    b_quality = self._predict_quality(
                        self.bottom_interpreter,
                        self.bottom_input_details,
                        self.bottom_output_details,
                        preprocessed_frame
                    )
                    
                    # Update best frame index if current is best
                    if len(contig_t) == 0 or t_quality > max(contig_t):
                        current_best_t = i
                    if len(contig_b) == 0 or b_quality > max(contig_b):
                        current_best_b = i
                    
                    # Store quality scores
                    contig_t.append(t_quality)
                    contig_b.append(b_quality)
                    
                elif in_contig:
                    # Was in segment but current frame has no crustacean
                    in_contig = False
                    
                    # Save best frame for this segment
                    best_frames[0].append(current_best_t)
                    best_frames[1].append(current_best_b)
                    
                    self.logger.debug(
                        f"Segment ended: top={current_best_t}, bottom={current_best_b}"
                    )
                    
                    # Reset variables for next segment
                    current_best_t = None
                    current_best_b = None
                    contig_t = []
                    contig_b = []
                
                # Read next frame
                success, image = video.read()
                if not success and i < total_frames - 1:
                    self.logger.warning(f"Failed to read frame {i+1}")
            
            # Handle case where video ends while in a segment
            if in_contig:
                best_frames[0].append(current_best_t)
                best_frames[1].append(current_best_b)
                self.logger.debug(
                    f"Final segment: top={current_best_t}, bottom={current_best_b}"
                )
            
            self.logger.info(
                f"Selected {len(best_frames[0])} segments with best frames"
            )
            
            return best_frames
            
        except Exception as e:
            raise InferenceError(
                f"Frame selection failed: {str(e)}",
                details={'model': 'FrameSelector', 'error': str(e)}
            ) from e
    
    def unload(self) -> None:
        """
        Release both model resources.
        """
        self.top_interpreter = None
        self.bottom_interpreter = None
        self.top_input_details = None
        self.top_output_details = None
        self.bottom_input_details = None
        self.bottom_output_details = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.logger.info("FrameSelector models unloaded")
    
    def is_loaded(self) -> bool:
        """
        Check if both models are loaded.
        
        Returns:
            True if both models are loaded, False otherwise
        """
        return (self.top_interpreter is not None and 
                self.bottom_interpreter is not None)
