"""
Unit tests for OfflinePipeline class.

Tests the OfflinePipeline including initialization, video file discovery,
completed files tracking, and the 4-stage processing pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from crustacean.core.offline_pipeline import OfflinePipeline
from crustacean.utils.config import Config


@pytest.fixture
def offline_pipeline_config_dict():
    """Configuration for OfflinePipeline testing."""
    return {
        'models': {
            'binary_classifier': {'path': 'test/bc.tflite'},
            'frame_selector': {
                'top_model_path': 'test/top.tflite',
                'bottom_model_path': 'test/bottom.tflite'
            },
            'object_detector': {'path': 'test/od.tflite'},
            'keypoint_detector': {'path': 'test/kd.tflite'}
        },
        'output': {
            'completed_files': './completed.txt',
            'extracted_frames_dir': './frames',
            'detections_dir': './detections'
        },
        'logging': {'level': 'INFO', 'console': True}
    }


@pytest.fixture
def offline_pipeline_config(temp_dir, offline_pipeline_config_dict):
    """Create Config object for testing."""
    import yaml
    config_path = temp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(offline_pipeline_config_dict, f)
    return Config.load(str(config_path))


class TestOfflinePipelineInitialization:
    """Test OfflinePipeline initialization."""
    
    def test_init_sets_video_dir(self, offline_pipeline_config, temp_dir):
        """Test that initialization sets video directory."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        
        assert pipeline.video_dir == video_dir
    
    def test_init_sets_config_paths(self, offline_pipeline_config, temp_dir):
        """Test that initialization sets paths from config."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        
        assert pipeline.completed_files_path == Path('./completed.txt')
        assert pipeline.extracted_frames_dir == Path('./frames')
        assert pipeline.output_dir == Path('./detections')
    
    def test_init_accepts_profiler(self, offline_pipeline_config, temp_dir):
        """Test that initialization accepts profiler."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        mock_profiler = Mock()
        
        pipeline = OfflinePipeline(
            offline_pipeline_config, 
            str(video_dir), 
            profiler=mock_profiler
        )
        
        assert pipeline.profiler is mock_profiler


class TestVideoFileDiscovery:
    """Test video file discovery."""
    
    def test_get_video_files_finds_mp4(self, offline_pipeline_config, temp_dir):
        """Test that _get_video_files finds .mp4 files."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        (video_dir / 'test1.mp4').touch()
        (video_dir / 'test2.mp4').touch()
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        files = pipeline._get_video_files()
        
        assert len(files) == 2
    
    def test_get_video_files_ignores_non_video(self, offline_pipeline_config, temp_dir):
        """Test that _get_video_files ignores non-video files."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        (video_dir / 'test.mp4').touch()
        (video_dir / 'readme.txt').touch()
        (video_dir / 'data.json').touch()
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        files = pipeline._get_video_files()
        
        assert len(files) == 1
        assert files[0].name == 'test.mp4'
    
    def test_get_video_files_empty_dir(self, offline_pipeline_config, temp_dir):
        """Test _get_video_files with empty directory."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        files = pipeline._get_video_files()
        
        assert len(files) == 0
    
    def test_get_video_files_nonexistent_dir(self, offline_pipeline_config, temp_dir):
        """Test _get_video_files with nonexistent directory."""
        pipeline = OfflinePipeline(
            offline_pipeline_config, 
            str(temp_dir / 'nonexistent')
        )
        files = pipeline._get_video_files()
        
        assert len(files) == 0


class TestCompletedFilesTracking:
    """Test completed files tracking."""
    
    def test_load_completed_files_empty(self, offline_pipeline_config, temp_dir):
        """Test loading when no completed files exist."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        pipeline.completed_files_path = temp_dir / 'completed.txt'
        
        completed = pipeline._load_completed_files()
        
        assert completed == set()
    
    def test_load_completed_files_existing(self, offline_pipeline_config, temp_dir):
        """Test loading existing completed files."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        
        completed_file = temp_dir / 'completed.txt'
        completed_file.write_text("video1.mp4\nvideo2.mp4\n")
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        pipeline.completed_files_path = completed_file
        
        completed = pipeline._load_completed_files()
        
        assert completed == {'video1.mp4', 'video2.mp4'}
    
    def test_mark_completed(self, offline_pipeline_config, temp_dir):
        """Test marking a file as completed."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        
        completed_file = temp_dir / 'completed.txt'
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        pipeline.completed_files_path = completed_file
        
        pipeline._mark_completed('test.mp4')
        
        assert 'test.mp4' in completed_file.read_text()


class TestPipelineRun:
    """Test pipeline run method."""
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_run_no_videos(
        self, mock_kd, mock_od, mock_fs, mock_bc, 
        offline_pipeline_config, temp_dir
    ):
        """Test run with no video files."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        pipeline.run()
        
        # Should not load models if no videos
        mock_bc.assert_not_called()
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_run_skips_completed(
        self, mock_kd, mock_od, mock_fs, mock_bc,
        offline_pipeline_config, temp_dir
    ):
        """Test that run skips already completed files."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        (video_dir / 'test.mp4').touch()
        
        completed_file = temp_dir / 'completed.txt'
        completed_file.write_text("test.mp4\n")
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        pipeline.completed_files_path = completed_file
        
        pipeline.run()
        
        # Should not load models if all files completed
        mock_bc.assert_not_called()


class TestFrameExtraction:
    """Test frame extraction functionality."""
    
    def test_prepare_frames_dir_creates(self, offline_pipeline_config, temp_dir):
        """Test that _prepare_frames_dir creates directory."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        frames_dir = temp_dir / 'frames'
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        pipeline.extracted_frames_dir = frames_dir
        
        pipeline._prepare_frames_dir()
        
        assert frames_dir.exists()
    
    def test_prepare_frames_dir_clears_existing(self, offline_pipeline_config, temp_dir):
        """Test that _prepare_frames_dir clears existing directory."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        frames_dir = temp_dir / 'frames'
        frames_dir.mkdir()
        (frames_dir / 'old_frame.png').touch()
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        pipeline.extracted_frames_dir = frames_dir
        
        pipeline._prepare_frames_dir()
        
        assert frames_dir.exists()
        assert len(list(frames_dir.iterdir())) == 0


class TestResultsSaving:
    """Test results saving functionality."""
    
    def test_save_results_creates_csv(self, offline_pipeline_config, temp_dir):
        """Test that _save_results creates CSV file."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        output_dir = temp_dir / 'output'
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        pipeline.output_dir = output_dir
        
        keypoints = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        ])
        
        pipeline._save_results(Path('test_video.mp4'), keypoints)
        
        output_file = output_dir / 'test_video_keypoints.csv'
        assert output_file.exists()
    
    def test_save_results_csv_format(self, offline_pipeline_config, temp_dir):
        """Test that CSV has correct format."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        output_dir = temp_dir / 'output'
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        pipeline.output_dir = output_dir
        
        keypoints = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
        
        pipeline._save_results(Path('test.mp4'), keypoints)
        
        output_file = output_dir / 'test_keypoints.csv'
        content = output_file.read_text()
        
        # Check header
        assert 'frame_idx' in content
        assert 'x0' in content
        assert 'y6' in content
    
    def test_save_results_empty_keypoints(self, offline_pipeline_config, temp_dir):
        """Test _save_results with empty keypoints."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        output_dir = temp_dir / 'output'
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        pipeline.output_dir = output_dir
        
        # Should not raise error
        pipeline._save_results(Path('test.mp4'), np.array([]))
        
        # Should not create file
        assert not (output_dir / 'test_keypoints.csv').exists()


class TestContextManager:
    """Test context manager functionality."""
    
    @patch('crustacean.models.binary_classifier.BinaryClassifier')
    @patch('crustacean.models.frame_selector.FrameSelector')
    @patch('crustacean.models.object_detector.ObjectDetector')
    @patch('crustacean.models.keypoint_detector.KeypointDetector')
    def test_context_manager_cleanup(
        self, mock_kd, mock_od, mock_fs, mock_bc,
        offline_pipeline_config, temp_dir
    ):
        """Test that context manager calls cleanup."""
        video_dir = temp_dir / 'videos'
        video_dir.mkdir()
        
        pipeline = OfflinePipeline(offline_pipeline_config, str(video_dir))
        
        with pipeline:
            pass
        
        assert pipeline.models == {}
