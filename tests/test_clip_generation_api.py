"""Tests for clip generation API endpoints."""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    from app import app
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = '/tmp/test_uploads'
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_video_cache():
    """Mock video cache for testing."""
    from app import video_cache
    task_id = 'test_task_123'
    video_cache[task_id] = '/tmp/test_video.mp4'
    yield task_id
    # Cleanup
    if task_id in video_cache:
        del video_cache[task_id]


class TestClipsGenerateEndpoint:
    """Tests for /api/clips/generate endpoint."""
    
    def test_generate_clips_requires_task_id(self, client):
        """Test that generate clips endpoint requires task_id."""
        response = client.post(
            '/api/clips/generate',
            data=json.dumps({'events': []}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'task_id' in data['error'].lower()
    
    def test_generate_clips_requires_events(self, client, mock_video_cache):
        """Test that generate clips endpoint requires events."""
        response = client.post(
            '/api/clips/generate',
            data=json.dumps({'task_id': mock_video_cache}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_generate_clips_validates_task_id(self, client):
        """Test that generate clips validates task_id exists."""
        response = client.post(
            '/api/clips/generate',
            data=json.dumps({
                'task_id': 'nonexistent_task',
                'events': [{'timestamp': 10, 'type': 'goal'}]
            }),
            content_type='application/json'
        )
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'not found' in data['error'].lower()
    
    @patch('app.Path')
    @patch('app.prepare_clips')
    def test_generate_clips_success(self, mock_prepare_clips, mock_path, client, mock_video_cache):
        """Test successful clip generation."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_prepare_clips.return_value = [
            '/tmp/clips/clip_000_goal_10.mp4',
            '/tmp/clips/clip_001_goal_20.mp4'
        ]
        
        events = [
            {'timestamp': 10, 'type': 'goal', 'description': 'First goal'},
            {'timestamp': 20, 'type': 'goal', 'description': 'Second goal'}
        ]
        
        response = client.post(
            '/api/clips/generate',
            data=json.dumps({
                'task_id': mock_video_cache,
                'events': events
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'clips' in data
        assert len(data['clips']) == 2
        assert data['total'] == 2
        
        # Verify clip info structure
        for clip_info in data['clips']:
            assert 'filename' in clip_info
            assert 'timestamp' in clip_info
            assert 'event_type' in clip_info
    
    @patch('app.Path')
    @patch('app.prepare_clips')
    def test_generate_clips_formats_timestamps(self, mock_prepare_clips, mock_path, client, mock_video_cache):
        """Test that timestamps are formatted correctly (mm:ss)."""
        mock_path.return_value.exists.return_value = True
        mock_prepare_clips.return_value = ['/tmp/clips/clip_000_goal_65.mp4']
        
        events = [{'timestamp': 65, 'type': 'goal'}]  # 1 minute 5 seconds
        
        response = client.post(
            '/api/clips/generate',
            data=json.dumps({
                'task_id': mock_video_cache,
                'events': events
            }),
            content_type='application/json'
        )
        
        data = json.loads(response.data)
        assert data['clips'][0]['timestamp'] == '1:05'


class TestClipsDownloadEndpoint:
    """Tests for /api/clips/download/<filename> endpoint."""
    
    def test_download_clip_not_found(self, client):
        """Test downloading non-existent clip returns 404."""
        response = client.get('/api/clips/download/nonexistent.mp4')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'not found' in data['error'].lower()
    
    @patch('app.Path')
    @patch('app.send_file')
    def test_download_clip_success(self, mock_send_file, mock_path, client):
        """Test successful clip download."""
        mock_path.return_value.exists.return_value = True
        mock_send_file.return_value = MagicMock()
        
        response = client.get('/api/clips/download/test_clip.mp4')
        
        # Verify send_file was called
        assert mock_send_file.called
        call_args = mock_send_file.call_args
        assert 'test_clip.mp4' in str(call_args[0][0])
        assert call_args[1]['as_attachment'] is True
        assert call_args[1]['download_name'] == 'test_clip.mp4'


class TestClipsConcatenateEndpoint:
    """Tests for /api/clips/concatenate endpoint."""
    
    def test_concatenate_requires_clips(self, client):
        """Test that concatenate endpoint requires clips list."""
        response = client.post(
            '/api/clips/concatenate',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'clips' in data['error'].lower()
    
    def test_concatenate_empty_clips_list(self, client):
        """Test concatenate with empty clips list."""
        response = client.post(
            '/api/clips/concatenate',
            data=json.dumps({'clips': []}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    @patch('app.Path')
    def test_concatenate_nonexistent_clips(self, mock_path, client):
        """Test concatenate with clips that don't exist."""
        mock_path.return_value.exists.return_value = False
        
        response = client.post(
            '/api/clips/concatenate',
            data=json.dumps({'clips': ['nonexistent1.mp4', 'nonexistent2.mp4']}),
            content_type='application/json'
        )
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'no valid clips' in data['error'].lower()
    
    @patch('src.video_tools.concatenate_clips')
    @patch('app.Path')
    def test_concatenate_clips_success(self, mock_path, mock_concatenate, client):
        """Test successful clip concatenation."""
        mock_path.return_value.exists.return_value = True
        mock_concatenate.return_value = True
        
        response = client.post(
            '/api/clips/concatenate',
            data=json.dumps({'clips': ['clip1.mp4', 'clip2.mp4']}),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'filename' in data
        assert data['filename'].startswith('highlight_reel_')
        assert data['filename'].endswith('.mp4')
        assert data['clips_count'] == 2
    
    @patch('src.video_tools.concatenate_clips')
    @patch('app.Path')
    def test_concatenate_clips_failure(self, mock_path, mock_concatenate, client):
        """Test concatenate when underlying function fails."""
        mock_path.return_value.exists.return_value = True
        mock_concatenate.return_value = False
        
        response = client.post(
            '/api/clips/concatenate',
            data=json.dumps({'clips': ['clip1.mp4', 'clip2.mp4']}),
            content_type='application/json'
        )
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'failed' in data['error'].lower()


class TestClipsMethodsEndpoint:
    """Tests for /api/clips/methods endpoint."""
    
    def test_get_clipping_methods(self, client):
        """Test retrieving available clipping methods."""
        response = client.get('/api/clips/methods')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'available_methods' in data
        assert 'has_ffmpeg' in data
        assert 'has_moviepy' in data
        assert 'can_clip' in data
        
        assert isinstance(data['available_methods'], list)
        assert isinstance(data['has_ffmpeg'], bool)
        assert isinstance(data['has_moviepy'], bool)
        assert isinstance(data['can_clip'], bool)
    
    def test_clipping_methods_structure(self, client):
        """Test that clipping methods response has correct structure."""
        response = client.get('/api/clips/methods')
        data = json.loads(response.data)
        
        # Should have at least one method available
        assert len(data['available_methods']) > 0 or not data['can_clip']
        
        # Logical consistency checks
        if data['has_ffmpeg']:
            assert any('ffmpeg' in method for method in data['available_methods'])
        
        if data['has_moviepy']:
            assert 'moviepy' in data['available_methods']
        
        if data['can_clip']:
            assert len(data['available_methods']) > 0


class TestClipGenerationIntegration:
    """Integration tests for clip generation workflow."""
    
    @patch('app.Path')
    @patch('app.prepare_clips')
    @patch('src.video_tools.concatenate_clips')
    def test_full_clip_generation_workflow(self, mock_concat, mock_prepare, mock_path, client, mock_video_cache):
        """Test full workflow: generate clips -> concatenate -> download."""
        mock_path.return_value.exists.return_value = True
        mock_prepare.return_value = [
            '/tmp/clips/clip_000_goal_10.mp4',
            '/tmp/clips/clip_001_goal_20.mp4'
        ]
        mock_concat.return_value = True
        
        events = [
            {'timestamp': 10, 'type': 'goal'},
            {'timestamp': 20, 'type': 'goal'}
        ]
        
        # Step 1: Generate clips
        gen_response = client.post(
            '/api/clips/generate',
            data=json.dumps({
                'task_id': mock_video_cache,
                'events': events
            }),
            content_type='application/json'
        )
        assert gen_response.status_code == 200
        gen_data = json.loads(gen_response.data)
        clips = [clip['filename'] for clip in gen_data['clips']]
        
        # Step 2: Concatenate clips
        concat_response = client.post(
            '/api/clips/concatenate',
            data=json.dumps({'clips': clips}),
            content_type='application/json'
        )
        assert concat_response.status_code == 200
        concat_data = json.loads(concat_response.data)
        assert concat_data['success'] is True
        
        # Verify workflow completed
        assert len(clips) == 2
        assert concat_data['clips_count'] == 2
    
    @patch('app.Path')
    @patch('app.prepare_clips')
    def test_selected_events_only(self, mock_prepare, mock_path, client, mock_video_cache):
        """Test that only selected events generate clips."""
        mock_path.return_value.exists.return_value = True
        mock_prepare.return_value = ['/tmp/clips/clip_000_goal_10.mp4']
        
        # Send only one event (simulating user selection)
        events = [{'timestamp': 10, 'type': 'goal'}]
        
        response = client.post(
            '/api/clips/generate',
            data=json.dumps({
                'task_id': mock_video_cache,
                'events': events
            }),
            content_type='application/json'
        )
        
        data = json.loads(response.data)
        assert len(data['clips']) == 1
        
        # Verify prepare_clips was called with exactly one event
        assert mock_prepare.called
        call_args = mock_prepare.call_args[0]
        assert len(call_args[0]) == 1  # First argument is events list
