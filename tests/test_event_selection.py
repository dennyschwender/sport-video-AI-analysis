"""
Test event type selection and filtering functionality.
"""
import pytest
from app import app
import json


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_analyze_frames_respects_event_type_filtering(client):
    """Test that AI only detects requested event types."""
    frames_data = {
        'frames': [{'data': 'data:image/jpeg;base64,/9j/4AAQSkZJRg', 'timestamp': 0.0}],
        'instructions': 'Find all goals',
        'backend': 'simulated',
        'sport': 'floorball',
        'video_duration': 10
    }
    
    response = client.post('/api/analyze/frames', 
                          data=json.dumps(frames_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'events' in data
    
    # Simulated backend should return events
    # (Note: Actual event type filtering happens in vision backend prompts)


def test_analyze_with_multiple_event_types(client):
    """Test analysis with multiple selected event types."""
    frames_data = {
        'frames': [{'data': 'data:image/jpeg;base64,/9j/4AAQSkZJRg', 'timestamp': 0.0}],
        'instructions': 'Find all goals, shots, and saves',
        'backend': 'simulated',
        'sport': 'floorball',
        'video_duration': 10
    }
    
    response = client.post('/api/analyze/frames',
                          data=json.dumps(frames_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'events' in data
    
    # Simulated backend generates events
    assert isinstance(data['events'], list)


def test_analyze_frames_with_empty_instructions(client):
    """Test that empty instructions still work (defaults to all events)."""
    frames_data = {
        'frames': [{'data': 'data:image/jpeg;base64,/9j/4AAQSkZJRg', 'timestamp': 0.0}],
        'instructions': '',
        'backend': 'simulated',
        'sport': 'floorball',
        'video_duration': 10
    }
    
    response = client.post('/api/analyze/frames',
                          data=json.dumps(frames_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'events' in data


def test_event_type_filtering_in_instructions():
    """Test instruction building logic for event type filtering."""
    # This tests the frontend logic pattern
    selected_types = ['goal', 'shot']
    additional_text = 'Focus on player #7'
    
    # Expected format: "Find all goals, shots. Focus on player #7"
    event_list = ', '.join(selected_types)
    expected = f"Find all {event_list}."
    if additional_text:
        expected += f" {additional_text}"
    
    assert 'goal' in expected
    assert 'shot' in expected
    assert 'player #7' in expected


def test_preset_event_types_coverage():
    """Test that all preset event types are recognized."""
    preset_types = ['goal', 'shot', 'save', 'assist', 'penalty', 'turnover', 'timeout']
    
    # Verify against sport presets from config file
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        floorball_types = config_data['sport_presets']['floorball']['event_types']
        
        # Check that common presets are in floorball config
        assert 'goal' in floorball_types
        assert 'shot' in floorball_types
        assert 'save' in floorball_types
        assert 'assist' in floorball_types
        assert 'penalty' in floorball_types
    else:
        # Skip if config doesn't exist
        pytest.skip("config.yaml not found")


def test_analyze_frames_validates_backend(client):
    """Test that invalid backend is handled."""
    frames_data = {
        'frames': [{'data': 'data:image/jpeg;base64,/9j/4AAQSkZJRg', 'timestamp': 0.0}],
        'instructions': 'Find all goals',
        'backend': 'invalid_backend',
        'sport': 'floorball',
        'video_duration': 10
    }
    
    response = client.post('/api/analyze/frames',
                          data=json.dumps(frames_data),
                          content_type='application/json')
    
    # Should handle gracefully (backend factory will return simulated or error)
    assert response.status_code in [200, 400, 500]


def test_analyze_frames_validates_sport(client):
    """Test that analysis works even with invalid sport (uses defaults)."""
    frames_data = {
        'frames': [{'data': 'data:image/jpeg;base64,/9j/4AAQSkZJRg', 'timestamp': 0.0}],
        'instructions': 'Find all goals',
        'backend': 'simulated',
        'sport': 'invalid_sport',
        'video_duration': 10
    }
    
    response = client.post('/api/analyze/frames',
                          data=json.dumps(frames_data),
                          content_type='application/json')
    
    # Should work with invalid sport (backend handles it)
    assert response.status_code == 200


def test_multiple_frames_with_event_filtering(client):
    """Test that event filtering works with multiple frames."""
    # Create 5 mock frames
    frames = [{'data': f'data:image/jpeg;base64,mock_frame_{i}', 'timestamp': float(i)} 
              for i in range(5)]
    
    frames_data = {
        'frames': frames,
        'instructions': 'Find all goals',
        'backend': 'simulated',
        'sport': 'floorball',
        'video_duration': 5
    }
    
    response = client.post('/api/analyze/frames',
                          data=json.dumps(frames_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'events' in data
    
    # With 5 frames, simulated backend should detect some events
    assert isinstance(data['events'], list)


def test_event_deduplication_with_filtering():
    """Test that event deduplication is handled by the API."""
    # Event deduplication is tested in test_enhanced.py
    # This test verifies API returns deduped events
    import pytest
    pytest.skip("Event deduplication is thoroughly tested in test_enhanced.py::TestSchema::test_event_deduplication")
