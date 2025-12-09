"""
Object Detector model for detecting and localizing crustaceans in frames.

This module implements the ObjectDetector model which detects crustaceans
in video frames and returns cropped ROI regions for keypoint detection.
"""

from typing import Tuple, Optional, Any
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite

from crustacean.models.base_model import BaseModel
from crustacean.utils.exceptions import ModelLoadError, InferenceError


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box format from center (x, y, w, h) to corner (x1, y1, x2, y2).
    
    Args:
        x: Array of bounding boxes in xywh format
        
    Returns:
        Array of bounding boxes in xyxy format
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Calculate Intersection over Union between a box and multiple boxes.
    
    Args:
        box: Single bounding box (x1, y1, x2, y2)
        boxes: Array of bounding boxes
        
    Returns:
        Array of IoU values
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    iou = intersection_area / (box_area + boxes_area - intersection_area)
    return iou


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 1,
) -> list:
    """
    Perform Non-Maximum Suppression on detection predictions.
    
    Args:
        prediction: Model output predictions
        conf_thres: Confidence threshold for filtering
        iou_thres: IoU threshold for NMS
        max_det: Maximum number of detections to return
        
    Returns:
        List of filtered detections per batch item
    """
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres

    max_nms = 30000
    output = [np.zeros((0, 6))] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        # Compute confidence
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])

        # Get class with max confidence
        conf = x[:, 5:].max(1)
        j = x[:, 5:].argmax(1)
        j = j.reshape(-1, 1)
        result = np.concatenate((box, conf.reshape(-1, 1), j.astype(float)), axis=1)
        conf_mask = conf > conf_thres
        x = result[conf_mask]

        n = x.shape[0]
        if not n:
            continue

        # Sort by confidence
        sorted_indices = np.argsort(x[:, 4])[::-1]
        x = x[sorted_indices][:max_nms]

        # NMS
        selected_indices = []
        for i in range(len(x)):
            iou = calculate_iou(x[i], x)
            iou[i] = 0  # Ignore self-iou
            mask = iou <= iou_thres
            x[i, 4] *= np.prod(mask)

            if x[i, 4] > 0:
                selected_indices.append(i)

        selected_indices = selected_indices[:max_det]
        output[xi] = x[selected_indices]

    return output


class ObjectDetector(BaseModel):
    """
    Detects and localizes crustaceans in video frames.
    
    This model processes video frames to detect crustaceans and returns
    cropped ROI regions suitable for keypoint detection. It uses a YOLO-style
    object detection model with Non-Maximum Suppression.
    
    The model expects:
    - A BGR frame from video capture
    
    Returns:
    - Cropped ROI region (grayscale)
    - Detection confidence score
    - Class index (0=crab, 1=lobster)
    
    Attributes:
        input_size: Model input size (default 640x640)
        confidence_threshold: Minimum confidence for valid detection
        fixed_crop_width: Width of fixed-size crop region
        fixed_crop_height: Height of fixed-size crop region
        
    Example:
        >>> config = Config.load()
        >>> od = ObjectDetector(config, preload=True)
        >>> roi, confidence, class_idx = od.predict(frame)
        >>> if confidence >= 0.75:
        ...     # Process ROI with keypoint detector
    """
    
    def __init__(self, config, preload: bool = False):
        """
        Initialize ObjectDetector with configuration.
        
        Args:
            config: Configuration object with model settings
            preload: If True, load model immediately
        """
        # Initialize base class without preloading
        super().__init__(config, preload=False)
        
        # Get configuration
        self.input_size = self.config.get('models.object_detector.input_size', 640)
        self.confidence_threshold = self.config.get(
            'models.object_detector.confidence_threshold', 0.75
        )
        self.fixed_crop_width = self.config.get(
            'models.object_detector.fixed_crop_width', 539
        )
        self.fixed_crop_height = self.config.get(
            'models.object_detector.fixed_crop_height', 561
        )
        
        if preload:
            self.load()
    
    def load(self) -> None:
        """
        Load the TFLite object detection model.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            model_path = self.config.get('models.object_detector.path')
            if not model_path:
                raise ModelLoadError(
                    "Object detector model path not found in configuration",
                    details={'config_key': 'models.object_detector.path'}
                )
            
            self.logger.info(f"Loading object detector from {model_path}")
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.logger.info("ObjectDetector model loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load ObjectDetector model: {str(e)}",
                details={'error': str(e)}
            ) from e
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess frame for object detection.
        
        Pads frame to square (1280x1280), then resizes to model input size.
        
        Args:
            frame: BGR frame from video (H, W, 3)
            
        Returns:
            Tuple of (preprocessed input, original frame copy)
        """
        # Keep copy of original for cropping
        true_scale_image = frame.copy()
        
        # Pad to 1280x1280 square
        fill_color = (0, 0, 0, 255)
        h, w, c = true_scale_image.shape
        target_size = 1280
        
        new_im = Image.new('RGBA', (target_size, target_size), fill_color)
        pos_y = (target_size - h) // 2
        new_im.paste(Image.fromarray(np.uint8(true_scale_image)), (0, pos_y))
        expanded_image = np.array(new_im)[..., :3]
        
        # Resize to model input size
        modified_image = cv2.resize(
            expanded_image, 
            (self.input_size, self.input_size)
        )
        
        # Reshape for model input
        input_data = np.reshape(
            modified_image, 
            (1, self.input_size, self.input_size, 3)
        )
        
        return input_data, true_scale_image
    
    def postprocess(self, output_data: np.ndarray) -> Tuple[float, float, float, float, float, int]:
        """
        Postprocess model output to get detection coordinates.
        
        Args:
            output_data: Raw model output
            
        Returns:
            Tuple of (x1, y1, x2, y2, confidence, class_index)
        """
        # This is handled in predict() due to complex multi-output handling
        pass
    
    def _crop_roi(
        self, 
        frame: np.ndarray, 
        y1: int, 
        y2: int
    ) -> np.ndarray:
        """
        Crop fixed-size ROI from frame at detection location.
        
        Args:
            frame: Original BGR frame
            y1: Y coordinate from detection
            y2: X coordinate from detection (naming from original code)
            
        Returns:
            Cropped grayscale ROI of fixed size
        """
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create fixed-size crop
        crop = np.zeros((self.fixed_crop_width, self.fixed_crop_height))
        
        for i in range(crop.shape[0]):
            for j in range(crop.shape[1]):
                ii, jj = y1 + i, y2 + j
                if ii < gray_frame.shape[0] and jj < gray_frame.shape[1]:
                    crop[i][j] = gray_frame[ii][jj]
        
        return crop
    
    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Detect crustacean in frame and return cropped ROI.
        
        Args:
            frame: BGR frame from video capture
            
        Returns:
            Tuple of (cropped_roi, confidence, class_index)
            - cropped_roi: Grayscale ROI array of shape (fixed_crop_width, fixed_crop_height)
            - confidence: Detection confidence score (0.0 to 1.0)
            - class_index: Class index (0=crab, 1=lobster)
            
        Raises:
            InferenceError: If detection fails
            
        Example:
            >>> roi, conf, cls = od.predict(frame)
            >>> if conf >= 0.75:
            ...     keypoints = kd.predict(roi)
        """
        if self.interpreter is None:
            raise InferenceError(
                "ObjectDetector model not loaded",
                details={'model': 'ObjectDetector'}
            )
        
        try:
            # Preprocess
            input_data, true_scale_image = self.preprocess(frame)
            
            # Get dimensions for scaling
            _, h, w, _ = input_data.shape
            
            # Run inference
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                input_data.astype(np.float32)
            )
            self.interpreter.invoke()
            
            # Get outputs
            y = []
            for output in self.output_details:
                y.append(self.interpreter.get_tensor(output['index']))
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            
            # Scale coordinates to pixel values
            y[0][..., :4] *= [w, h, w, h]
            
            # Apply NMS
            detections = non_max_suppression(y[0])
            
            if len(detections[0]) == 0:
                self.logger.warning("No detections found in frame")
                # Return empty ROI with zero confidence
                empty_roi = np.zeros((self.fixed_crop_width, self.fixed_crop_height))
                return empty_roi, 0.0, -1
            
            # Get best detection
            x1, y1, x2, y2, conf, class_index = detections[0][0]
            
            # Log confidence
            if conf < self.confidence_threshold:
                self.logger.warning(f"Low confidence detection: {conf:.3f}")
            
            self.logger.debug(f"Detection: class={int(class_index)}, conf={conf:.3f}")
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Crop ROI from original frame
            roi = self._crop_roi(true_scale_image, y1, y2)
            
            return roi, float(conf), int(class_index)
            
        except Exception as e:
            raise InferenceError(
                f"Object detection failed: {str(e)}",
                details={'model': 'ObjectDetector', 'error': str(e)}
            ) from e
