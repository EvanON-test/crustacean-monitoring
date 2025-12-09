"""
Integration tests for OfflinePipeline.

Tests the complete offline pipeline processing flow including:
- End-to-end video processing through all 4 stages
- Output file creation and format verification
- Completed files tracking
- Error handling during processing
"""

import pytest
import numpy as np
import csv
import cv2
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import yaml

from crustacean.core.offline_pipeline import OfflinePipeline
from crustacean.utils.config import Config


@pytest.fixture
def integration_config_dict():
    """Configuration for integration testing."""
    return {
        'models': {
            'binary_classifier': {
                'path': 'test/bc_model.tflite',
                'input_width': 320,
                'input_height': 180,
                'smoothing_gamma': 20,
                'rectify_theta': 0.5
            },
            'frame_selector': {
                'top_model_path': 'test/top_model.tflite',
                'bottom_model_path': 'test/bottom_model.tflite',
                'input_width': 320,
                'input_height': 180
            },
            'object_detector': {
                'path': 'test/od_model.tflite',
                'input_size': 640,
                'confidence_threshold': 0.75,
                'fixed_crop_width': 100,
                'fixed_crop_height': 100
            },
            'keypoint_detector': {
                'path': 'test/kd_model.tflite',
                'num_keypoints': 7
            }
        },
        'output': {
            'completed_files': './completed.txt',
            'extracted_frames_dir': './frames',
            'detections_dir': './detections'
        },
        'logging': {
            'level': 'DEBUG',
            'console': True
        }
    }


@pytest.fixture
def integration_config(temp_dir, integration_config_dict):
    """Create Config object for integration testing."""
    config_path = temp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(integration_config_dict, f)
    return Config.load(str(config_path))


@pytest.fixture
def test_video(temp_dir):
    """
    Create a test video file with known frame count.
    
    Creates a simple 10-frame video with colored frames for testing.
    """
    video_dir = temp_dir / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / 'test_video.mp4'
    
    # Create a simple test video with 10 frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    for i in range(10):
        # Create frames with different colors for each frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, i % 3] = 100 + i * 15  # Vary color intensity
        writer.write(frame)
    
    writer.release()
    
    return video_path


@pytest.fixture
def mock_models():
    """
    Create mock model instances with predictable outputs.
    
    Returns a dictionary of mock model instances.
    """
    # Mock Binary Classifier
    mock_bc = MagicMock()
    mock_bc.is_loaded.return_value = False
    mock_bc.predict.return_value = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0])
    
    # Mock Frame Selector
    mock_fs = MagicMock()
    mock_fs.is_loaded.return_value = False
    mock_fs.predict.return_value = [[2, 5], [2, 6]]  # [top_indices, bottom_indices]
    
    # Mock Object Detector - returns ROI, confidence, class_index
    mock_od = MagicMock()
    mock_od.is_loaded.return_value = False
    # Return a valid ROI (100x100 image), confidence, and class
    mock_roi = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mock_od.predict.return_value = (mock_roi, 0.85, 0)
    
    # Mock Keypoint Detector
    mock_kd = MagicMock()
    mock_kd.is_loaded.return_value = False
    # Return keypoints for 2 frames (14 values each: 7 keypoints * 2 coords)
    mock_kd.predict.return_value = np.array([
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        [15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145]
    ])
    
    return {
        'bc': mock_bc,
        'fs': mock_fs,
        'od': mock_od,
        'kd': mock_kd
    }


class TestOfflinePipelineEndToEnd:
    """Test complete end-to-end pipeline processing."""
    
    @patch('crustacean.core.offline_pipeline.cv2.VideoCapture')
    @patch('crustacean.core.offline_pipeline.cv2.imread')
    @patch('crustacean.core.offline_pipeline.cv2.imwrite')
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_full_pipeline_processing(
        self, mock_kd_class, mock_od_class, mock_fs_class, mock_bc_class,
        mock_imwrite, mock_imread, mock_video_capture,
        integration_config, temp_dir, mock_models
    ):
        """Test complete pipeline processes video and creates outputs."""
        # Setup video directory with test video
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / 'test_video.mp4').touch()
        
        # Setup output directories
        output_dir = temp_dir / 'detections'
        frames_dir = temp_dir / 'frames'
        completed_file = temp_dir / 'completed.txt'
        
        # Configure mock video capture
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.return_value = 10  # 10 frames
        mock_video_capture.return_value = mock_cap
        
        # Configure mock imread to return valid frames
        mock_imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Configure model class mocks to return our mock instances
        mock_bc_class.return_value = mock_models['bc']
        mock_fs_class.return_value = mock_models['fs']
        mock_od_class.return_value = mock_models['od']
        mock_kd_class.return_value = mock_models['kd']
        
        # Make imwrite actually create files so glob finds them
        def create_frame_file(path, frame):
            Path(path).touch()
            return True
        mock_imwrite.side_effect = create_frame_file
        
        # Create and configure pipeline
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.output_dir = output_dir
        pipeline.extracted_frames_dir = frames_dir
        pipeline.completed_files_path = completed_file
        
        # Run pipeline
        pipeline.run()
        
        # Verify models were loaded
        mock_bc_class.assert_called_once()
        mock_fs_class.assert_called_once()
        mock_od_class.assert_called_once()
        mock_kd_class.assert_called_once()
        
        # Verify each model's predict was called
        mock_models['bc'].predict.assert_called()
        mock_models['fs'].predict.assert_called()
        mock_models['od'].predict.assert_called()
        mock_models['kd'].predict.assert_called()
        
        # Verify output file was created
        assert output_dir.exists()
        
        # Verify completed file was updated
        assert completed_file.exists()
        assert 'test_video.mp4' in completed_file.read_text()


class TestOutputFileCreation:
    """Test output file creation and format."""
    
    def test_keypoint_csv_created(self, integration_config, temp_dir):
        """Test that keypoint CSV file is created with correct name."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        output_dir = temp_dir / 'detections'
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.output_dir = output_dir
        
        # Create test keypoints
        keypoints = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        ])
        
        # Save results
        pipeline._save_results(Path('my_video.mp4'), keypoints)
        
        # Verify file exists with correct name
        expected_file = output_dir / 'my_video_keypoints.csv'
        assert expected_file.exists()
    
    def test_keypoint_csv_header_format(self, integration_config, temp_dir):
        """Test that CSV has correct header format."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        output_dir = temp_dir / 'detections'
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.output_dir = output_dir
        
        keypoints = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
        pipeline._save_results(Path('test.mp4'), keypoints)
        
        # Read and verify header
        output_file = output_dir / 'test_keypoints.csv'
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
        
        # Expected header: frame_idx, x0, y0, x1, y1, ..., x6, y6
        expected_header = ['frame_idx']
        for i in range(7):
            expected_header.extend([f'x{i}', f'y{i}'])
        
        assert header == expected_header
    
    def test_keypoint_csv_data_format(self, integration_config, temp_dir):
        """Test that CSV data rows have correct format."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        output_dir = temp_dir / 'detections'
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.output_dir = output_dir
        
        # Create keypoints with known values
        keypoints = np.array([
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
            [15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145]
        ])
        pipeline._save_results(Path('test.mp4'), keypoints)
        
        # Read and verify data
        output_file = output_dir / 'test_keypoints.csv'
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            rows = list(reader)
        
        # Verify row count
        assert len(rows) == 2
        
        # Verify first row data
        assert rows[0][0] == '0'  # frame_idx
        assert rows[0][1] == '10'  # x0
        assert rows[0][2] == '20'  # y0
        
        # Verify second row data
        assert rows[1][0] == '1'  # frame_idx
        assert rows[1][1] == '15'  # x0
    
    def test_keypoint_csv_multiple_frames(self, integration_config, temp_dir):
        """Test CSV with multiple frames of keypoints."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        output_dir = temp_dir / 'detections'
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.output_dir = output_dir
        
        # Create 5 frames of keypoints
        keypoints = np.random.randint(0, 500, (5, 14))
        pipeline._save_results(Path('multi_frame.mp4'), keypoints)
        
        # Verify all rows present
        output_file = output_dir / 'multi_frame_keypoints.csv'
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            rows = list(reader)
        
        assert len(rows) == 5
        
        # Verify frame indices are sequential
        for i, row in enumerate(rows):
            assert row[0] == str(i)


class TestCompletedFilesTracking:
    """Test completed files tracking functionality."""
    
    def test_completed_file_created_on_first_run(self, integration_config, temp_dir):
        """Test that completed file is created on first run."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        completed_file = temp_dir / 'completed.txt'
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.completed_files_path = completed_file
        
        # Mark a file as completed
        pipeline._mark_completed('video1.mp4')
        
        assert completed_file.exists()
        assert 'video1.mp4' in completed_file.read_text()
    
    def test_completed_files_appended(self, integration_config, temp_dir):
        """Test that completed files are appended, not overwritten."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        completed_file = temp_dir / 'completed.txt'
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.completed_files_path = completed_file
        
        # Mark multiple files
        pipeline._mark_completed('video1.mp4')
        pipeline._mark_completed('video2.mp4')
        pipeline._mark_completed('video3.mp4')
        
        content = completed_file.read_text()
        assert 'video1.mp4' in content
        assert 'video2.mp4' in content
        assert 'video3.mp4' in content
    
    def test_completed_files_loaded_correctly(self, integration_config, temp_dir):
        """Test that completed files are loaded correctly."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        completed_file = temp_dir / 'completed.txt'
        completed_file.write_text("video1.mp4\nvideo2.mp4\nvideo3.mp4\n")
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.completed_files_path = completed_file
        
        completed = pipeline._load_completed_files()
        
        assert len(completed) == 3
        assert 'video1.mp4' in completed
        assert 'video2.mp4' in completed
        assert 'video3.mp4' in completed
    
    def test_completed_files_skipped_in_run(self, integration_config, temp_dir):
        """Test that already completed files are skipped during run."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / 'completed_video.mp4').touch()
        (video_dir / 'new_video.mp4').touch()
        
        completed_file = temp_dir / 'completed.txt'
        completed_file.write_text("completed_video.mp4\n")
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.completed_files_path = completed_file
        
        # Get pending files
        video_files = pipeline._get_video_files()
        completed = pipeline._load_completed_files()
        pending = [f for f in video_files if f.name not in completed]
        
        # Only new_video.mp4 should be pending
        assert len(pending) == 1
        assert pending[0].name == 'new_video.mp4'
    
    def test_empty_completed_file_handled(self, integration_config, temp_dir):
        """Test that empty completed file is handled correctly."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        completed_file = temp_dir / 'completed.txt'
        completed_file.write_text("")
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.completed_files_path = completed_file
        
        completed = pipeline._load_completed_files()
        
        assert completed == set()
    
    def test_completed_file_with_whitespace(self, integration_config, temp_dir):
        """Test that whitespace in completed file is handled."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        completed_file = temp_dir / 'completed.txt'
        completed_file.write_text("video1.mp4\n\n  \nvideo2.mp4\n  video3.mp4  \n")
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.completed_files_path = completed_file
        
        completed = pipeline._load_completed_files()
        
        # Should handle whitespace-only lines
        assert 'video1.mp4' in completed
        assert 'video2.mp4' in completed


class TestVideoFileDiscoveryIntegration:
    """Integration tests for video file discovery."""
    
    def test_discovers_multiple_video_formats(self, integration_config, temp_dir):
        """Test that multiple video formats are discovered."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Create various video files
        (video_dir / 'video1.mp4').touch()
        (video_dir / 'video2.avi').touch()
        (video_dir / 'video3.mov').touch()
        (video_dir / 'video4.mkv').touch()
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        files = pipeline._get_video_files()
        
        assert len(files) == 4
        names = {f.name for f in files}
        assert 'video1.mp4' in names
        assert 'video2.avi' in names
        assert 'video3.mov' in names
        assert 'video4.mkv' in names
    
    def test_ignores_non_video_files(self, integration_config, temp_dir):
        """Test that non-video files are ignored."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mixed files
        (video_dir / 'video.mp4').touch()
        (video_dir / 'readme.txt').touch()
        (video_dir / 'data.json').touch()
        (video_dir / 'image.jpg').touch()
        (video_dir / 'script.py').touch()
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        files = pipeline._get_video_files()
        
        assert len(files) == 1
        assert files[0].name == 'video.mp4'
    
    def test_files_returned_sorted(self, integration_config, temp_dir):
        """Test that video files are returned in sorted order."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Create files in non-alphabetical order
        (video_dir / 'zebra.mp4').touch()
        (video_dir / 'alpha.mp4').touch()
        (video_dir / 'middle.mp4').touch()
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        files = pipeline._get_video_files()
        
        names = [f.name for f in files]
        assert names == ['alpha.mp4', 'middle.mp4', 'zebra.mp4']


class TestPipelineStageIntegration:
    """Test individual pipeline stages in integration context."""
    
    @patch('crustacean.core.offline_pipeline.cv2.VideoCapture')
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_binary_classifier_stage(
        self, mock_kd, mock_od, mock_fs, mock_bc, mock_video_capture,
        integration_config, temp_dir
    ):
        """Test binary classifier stage processes video correctly."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / 'test.mp4'
        video_path.touch()
        
        # Setup mock video capture
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # Setup mock BC
        mock_bc_instance = MagicMock()
        mock_bc_instance.is_loaded.return_value = False
        mock_bc_instance.predict.return_value = np.array([0, 1, 1, 0, 1])
        mock_bc.return_value = mock_bc_instance
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.load_models(preload=False)
        
        # Run BC stage
        signal = pipeline._run_binary_classifier(video_path)
        
        # Verify BC was called
        mock_bc_instance.load.assert_called_once()
        mock_bc_instance.predict.assert_called_once()
        mock_bc_instance.unload.assert_called_once()
        
        # Verify signal returned
        assert len(signal) == 5
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_keypoint_detector_stage(
        self, mock_kd, mock_od, mock_fs, mock_bc,
        integration_config, temp_dir
    ):
        """Test keypoint detector stage processes ROIs correctly."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mock KD
        mock_kd_instance = MagicMock()
        mock_kd_instance.is_loaded.return_value = False
        expected_keypoints = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        ])
        mock_kd_instance.predict.return_value = expected_keypoints
        mock_kd.return_value = mock_kd_instance
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.load_models(preload=False)
        
        # Create test ROI frames
        roi_frames = np.random.randint(0, 255, (2, 100, 100, 3), dtype=np.uint8)
        
        # Run KD stage
        keypoints = pipeline._run_keypoint_detector(roi_frames)
        
        # Verify KD was called
        mock_kd_instance.load.assert_called_once()
        mock_kd_instance.predict.assert_called_once()
        mock_kd_instance.unload.assert_called_once()
        
        # Verify keypoints returned
        assert keypoints.shape == (2, 14)
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_keypoint_detector_empty_input(
        self, mock_kd, mock_od, mock_fs, mock_bc,
        integration_config, temp_dir
    ):
        """Test keypoint detector handles empty input gracefully."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.load_models(preload=False)
        
        # Run KD stage with empty input
        keypoints = pipeline._run_keypoint_detector(np.array([]))
        
        # Should return empty array
        assert len(keypoints) == 0


class TestPipelineCleanup:
    """Test pipeline cleanup and resource management."""
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_cleanup_called_after_run(
        self, mock_kd, mock_od, mock_fs, mock_bc,
        integration_config, temp_dir
    ):
        """Test that cleanup is called after run completes."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mock models
        for mock_class in [mock_bc, mock_fs, mock_od, mock_kd]:
            instance = MagicMock()
            instance.is_loaded.return_value = True
            mock_class.return_value = instance
        
        pipeline = OfflinePipeline(integration_config, str(video_dir))
        pipeline.run()  # No videos, but should still cleanup
        
        # Models dict should be empty after cleanup
        assert pipeline.models == {}
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_context_manager_cleanup(
        self, mock_kd, mock_od, mock_fs, mock_bc,
        integration_config, temp_dir
    ):
        """Test that context manager properly cleans up."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mock models
        for mock_class in [mock_bc, mock_fs, mock_od, mock_kd]:
            instance = MagicMock()
            instance.is_loaded.return_value = True
            mock_class.return_value = instance
        
        with OfflinePipeline(integration_config, str(video_dir)) as pipeline:
            pipeline.load_models(preload=True)
            assert len(pipeline.models) == 4
        
        # After context exit, models should be cleaned up
        assert pipeline.models == {}
