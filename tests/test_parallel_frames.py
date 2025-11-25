"""Test parallel frame processing in /api/analyze/frames endpoint."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_parallel_processing_enabled_for_openai(client):
    """Test that parallel processing is used for OpenAI backend."""
    # Mock frames data
    frames_data = []
    for i in range(150):  # 3 chunks of 50 frames
        frames_data.append({
            'timestamp': i * 0.5,
            'data': 'data:image/jpeg;base64,/9j/4AAQSkZJRg=='
        })
    
    with patch('app.get_vision_backend') as mock_backend, \
         patch('app.config') as mock_config:
        
        # Configure mock backend
        mock_vision = Mock()
        mock_vision._analyze_frames_impl.return_value = [
            {'timestamp': '0:00', 'event_type': 'test', 'description': 'test event'}
        ]
        mock_backend.return_value = mock_vision
        
        # Configure rate limits
        mock_config.openai_rate_limit_rpm = 500
        mock_config.rate_limit_retry_delay = 0.1
        mock_config.rate_limit_max_retries = 3
        mock_config.openai_model = 'gpt-4o-mini'
        
        # Make request
        response = client.post('/api/analyze/frames', json={
            'frames': frames_data,
            'instructions': 'Find all goals',
            'backend': 'openai',
            'sport': 'floorball',
            'video_duration': 75.0
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        
        # Verify _analyze_frames_impl was called 3 times (3 chunks)
        assert mock_vision._analyze_frames_impl.call_count == 3


def test_sequential_processing_for_simulated(client):
    """Test that simulated backend uses sequential processing."""
    frames_data = []
    for i in range(100):
        frames_data.append({
            'timestamp': i * 0.5,
            'data': 'data:image/jpeg;base64,/9j/4AAQSkZJRg=='
        })
    
    with patch('app.get_vision_backend') as mock_backend, \
         patch('app.config') as mock_config:
        
        mock_vision = Mock()
        mock_vision._analyze_frames_impl.return_value = []
        mock_backend.return_value = mock_vision
        
        mock_config.rate_limit_retry_delay = 0.1
        mock_config.rate_limit_max_retries = 3
        
        response = client.post('/api/analyze/frames', json={
            'frames': frames_data,
            'instructions': 'Find all goals',
            'backend': 'simulated',
            'sport': 'floorball',
            'video_duration': 50.0
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True


def test_chunk_results_combined_in_order(client):
    """Test that chunk results are combined in correct order."""
    frames_data = []
    for i in range(150):
        frames_data.append({
            'timestamp': i * 0.5,
            'data': 'data:image/jpeg;base64,/9j/4AAQSkZJRg=='
        })
    
    with patch('app.get_vision_backend') as mock_backend, \
         patch('app.config') as mock_config:
        
        # Mock returns different events for each chunk
        mock_vision = Mock()
        call_count = [0]
        
        def mock_analyze(*args, **kwargs):
            call_count[0] += 1
            return [{'event_type': f'event_{call_count[0]}', 'timestamp': '0:00', 'description': f'Event {call_count[0]}'}]
        
        mock_vision._analyze_frames_impl.side_effect = mock_analyze
        mock_backend.return_value = mock_vision
        
        mock_config.openai_rate_limit_rpm = 500
        mock_config.rate_limit_retry_delay = 0.1
        mock_config.rate_limit_max_retries = 3
        mock_config.openai_model = 'gpt-4o-mini'
        
        response = client.post('/api/analyze/frames', json={
            'frames': frames_data,
            'instructions': 'Find all goals',
            'backend': 'openai',
            'sport': 'floorball',
            'video_duration': 75.0
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert len(data['events']) == 3  # 3 chunks
        
        # Events should be in order (chunk 1, 2, 3)
        assert data['events'][0]['event_type'] == 'event_1'
        assert data['events'][1]['event_type'] == 'event_2'
        assert data['events'][2]['event_type'] == 'event_3'
