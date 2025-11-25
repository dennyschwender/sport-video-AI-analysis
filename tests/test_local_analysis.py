"""Tests for local analysis functionality (frame-only upload)."""
import sys
from pathlib import Path
import json
import base64
from io import BytesIO
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def create_mock_frame(timestamp: float) -> dict:
    """Create a mock base64-encoded frame."""
    # Create a simple 10x10 pixel image
    try:
        from PIL import Image
        img = Image.new('RGB', (10, 10), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return {
            'timestamp': timestamp,
            'data': f'data:image/jpeg;base64,{img_data}'
        }
    except ImportError:
        # If PIL not available, create minimal base64 string
        mock_data = base64.b64encode(b'fake_image_data').decode('utf-8')
        return {
            'timestamp': timestamp,
            'data': f'data:image/jpeg;base64,{mock_data}'
        }


def test_local_analysis_page_loads(client):
    """Test that /local route returns the local analysis page."""
    response = client.get('/local')
    assert response.status_code == 200
    assert b'local' in response.data.lower()


def test_analyze_frames_endpoint_exists(client):
    """Test that /api/analyze/frames endpoint exists."""
    response = client.post('/api/analyze/frames', json={})
    # Should not be 404
    assert response.status_code != 404


def test_analyze_frames_requires_frames(client):
    """Test that analyze_frames returns error without frames."""
    response = client.post('/api/analyze/frames', 
                          json={
                              'instructions': 'Find all goals',
                              'backend': 'simulated',
                              'sport': 'floorball'
                          },
                          content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'frames' in data['error'].lower()


def test_analyze_frames_with_simulated_backend(client):
    """Test frame analysis with simulated backend."""
    frames = [
        create_mock_frame(0.0),
        create_mock_frame(8.0),
        create_mock_frame(16.0)
    ]
    
    response = client.post('/api/analyze/frames',
                          json={
                              'frames': frames,
                              'instructions': 'Find all goals and shots',
                              'backend': 'simulated',
                              'sport': 'floorball',
                              'video_duration': 24.0
                          },
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert data['success'] is True
    assert 'events' in data
    assert 'meta' in data
    assert isinstance(data['events'], list)
    assert data['meta']['frames_analyzed'] == 3
    assert data['meta']['video_duration'] == 24.0
    assert data['meta']['backend'] == 'simulated'


def test_analyze_frames_with_different_sports(client):
    """Test that different sports work correctly."""
    frames = [create_mock_frame(0.0), create_mock_frame(10.0)]
    
    for sport in ['floorball', 'hockey', 'soccer']:
        response = client.post('/api/analyze/frames',
                              json={
                                  'frames': frames,
                                  'instructions': 'Find all goals',
                                  'backend': 'simulated',
                                  'sport': sport,
                                  'video_duration': 20.0
                              },
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True


def test_analyze_frames_handles_single_frame(client):
    """Test that analysis works with a single frame."""
    frames = [create_mock_frame(0.0)]
    
    response = client.post('/api/analyze/frames',
                          json={
                              'frames': frames,
                              'instructions': 'Find all events',
                              'backend': 'simulated',
                              'sport': 'floorball',
                              'video_duration': 1.0
                          },
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] is True
    assert data['meta']['frames_analyzed'] == 1


def test_analyze_frames_handles_many_frames(client):
    """Test that analysis works with many frames."""
    # Create 25 frames
    frames = [create_mock_frame(i * 8.0) for i in range(25)]
    
    response = client.post('/api/analyze/frames',
                          json={
                              'frames': frames,
                              'instructions': 'Find all goals and penalties',
                              'backend': 'simulated',
                              'sport': 'floorball',
                              'video_duration': 200.0
                          },
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] is True
    assert data['meta']['frames_analyzed'] == 25


def test_analyze_frames_with_custom_instructions(client):
    """Test that custom instructions are accepted."""
    frames = [create_mock_frame(0.0), create_mock_frame(10.0)]
    
    custom_instructions = [
        'Find all goals and assists',
        'Detect saves and blocks',
        'Look for penalties and turnovers'
    ]
    
    for instruction in custom_instructions:
        response = client.post('/api/analyze/frames',
                              json={
                                  'frames': frames,
                                  'instructions': instruction,
                                  'backend': 'simulated',
                                  'sport': 'floorball',
                                  'video_duration': 20.0
                              },
                              content_type='application/json')
        
        assert response.status_code == 200


def test_analyze_frames_with_invalid_json(client):
    """Test that invalid JSON is handled gracefully."""
    response = client.post('/api/analyze/frames',
                          data='invalid json',
                          content_type='application/json')
    
    assert response.status_code in [400, 500]


def test_analyze_frames_with_missing_backend(client):
    """Test that missing backend defaults to simulated."""
    frames = [create_mock_frame(0.0)]
    
    response = client.post('/api/analyze/frames',
                          json={
                              'frames': frames,
                              'instructions': 'Find all goals',
                              'sport': 'floorball',
                              'video_duration': 10.0
                              # backend not specified
                          },
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['meta']['backend'] == 'simulated'


def test_analyze_frames_timestamps_are_preserved(client):
    """Test that frame timestamps are properly used in analysis."""
    frames = [
        create_mock_frame(0.0),
        create_mock_frame(15.5),
        create_mock_frame(32.0)
    ]
    
    response = client.post('/api/analyze/frames',
                          json={
                              'frames': frames,
                              'instructions': 'Find all events',
                              'backend': 'simulated',
                              'sport': 'floorball',
                              'video_duration': 40.0
                          },
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Events should have timestamps within video duration
    for event in data['events']:
        assert 0 <= event['timestamp'] <= 40.0


def test_analyze_frames_event_structure(client):
    """Test that returned events have proper structure."""
    frames = [create_mock_frame(0.0), create_mock_frame(10.0)]
    
    response = client.post('/api/analyze/frames',
                          json={
                              'frames': frames,
                              'instructions': 'Find all goals',
                              'backend': 'simulated',
                              'sport': 'floorball',
                              'video_duration': 20.0
                          },
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Check event structure
    for event in data['events']:
        assert 'timestamp' in event
        assert 'type' in event
        assert 'description' in event
        assert 'confidence' in event
        assert isinstance(event['timestamp'], (int, float))
        assert isinstance(event['type'], str)
        assert isinstance(event['confidence'], (int, float))
        assert 0 <= event['confidence'] <= 1
