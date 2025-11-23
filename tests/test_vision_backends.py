"""Tests for vision-capable backends."""
import sys
from pathlib import Path
import tempfile
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vision_backends import SimulatedVisionBackend, get_vision_backend


def test_simulated_vision_backend_generates_events():
    """Test that simulated vision backend generates events based on instructions."""
    backend = SimulatedVisionBackend()
    
    # Create a fake video file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    try:
        result = backend.analyze_video_frames(
            video_path=video_path,
            instructions='Find all goals and penalties',
            sport='floorball'
        )
        
        assert 'events' in result
        assert 'meta' in result
        assert isinstance(result['events'], list)
        
        # Should generate some events based on instructions
        events = result['events']
        if events:  # May have events
            for event in events:
                assert 'type' in event
                assert 'timestamp' in event
                assert 'description' in event
                assert 'confidence' in event
                assert event['type'] in ['goal', 'penalty', 'shot', 'save', 'turnover']
    
    finally:
        Path(video_path).unlink(missing_ok=True)


def test_simulated_vision_backend_respects_instructions():
    """Test that simulated backend generates events matching instructions."""
    backend = SimulatedVisionBackend()
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    try:
        result = backend.analyze_video_frames(
            video_path=video_path,
            instructions='Find all goals',
            sport='floorball'
        )
        
        events = result['events']
        if events:
            # Should generate goal events when asked for goals
            goal_events = [e for e in events if e['type'] == 'goal']
            assert len(goal_events) > 0, "Should generate goal events when instructed"
    
    finally:
        Path(video_path).unlink(missing_ok=True)


def test_get_vision_backend_factory():
    """Test vision backend factory function."""
    # Get simulated backend
    backend = get_vision_backend('simulated')
    assert isinstance(backend, SimulatedVisionBackend)
    
    # Without API key, should return simulated
    backend = get_vision_backend('openai', api_key=None)
    assert isinstance(backend, SimulatedVisionBackend)
    
    # With API key, should return OpenAI backend if openai package is installed
    try:
        import openai
        backend = get_vision_backend('openai', api_key='sk-test-key')
        assert backend is not None
    except ImportError:
        # OpenAI package not installed, skip this test
        pass


def test_vision_backend_handles_missing_video():
    """Test that vision backend handles missing video gracefully."""
    backend = SimulatedVisionBackend()
    
    result = backend.analyze_video_frames(
        video_path='/nonexistent/video.mp4',
        instructions='Find all goals',
        sport='floorball'
    )
    
    assert 'events' in result
    assert 'meta' in result
    # Should handle error gracefully
    if 'error' in result['meta']:
        assert isinstance(result['meta']['error'], str)


def test_simulated_vision_backend_returns_timestamps():
    """Test that events have proper timestamp values."""
    backend = SimulatedVisionBackend()
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    try:
        result = backend.analyze_video_frames(
            video_path=video_path,
            instructions='Find all goals and saves',
            sport='floorball'
        )
        
        events = result['events']
        for event in events:
            assert isinstance(event['timestamp'], (int, float))
            assert event['timestamp'] >= 0
            assert 0 <= event['confidence'] <= 1
    
    finally:
        Path(video_path).unlink(missing_ok=True)


def test_parse_events_from_text():
    """Test JSON event parsing from text."""
    from src.vision_backends import VisionBackendMixin
    
    text = '''
    Here are the events I found:
    {"timestamp": 12.5, "type": "goal", "description": "Goal scored", "confidence": 0.9}
    Some other text
    {"timestamp": 45.0, "type": "shot", "description": "Shot on goal", "confidence": 0.8}
    '''
    
    events = VisionBackendMixin.parse_events_from_text(text)
    
    assert len(events) == 2
    assert events[0]['type'] == 'goal'
    assert events[0]['timestamp'] == 12.5
    assert events[1]['type'] == 'shot'
    assert events[1]['timestamp'] == 45.0


def test_parse_events_handles_invalid_json():
    """Test that invalid JSON is skipped gracefully."""
    from src.vision_backends import VisionBackendMixin
    
    text = '''
    {"timestamp": 12.5, "type": "goal"}
    {invalid json}
    {"timestamp": 45.0, "type": "shot", "description": "test", "confidence": 0.8}
    '''
    
    events = VisionBackendMixin.parse_events_from_text(text)
    
    # Should skip invalid JSON but parse valid ones
    assert len(events) == 2


def test_gemini_vision_backend_initialization():
    """Test that Gemini backend initializes correctly."""
    try:
        import google.generativeai as genai
        from src.vision_backends import GeminiVisionBackend
        
        # Test initialization with API key
        backend = GeminiVisionBackend(api_key='test-key', model='gemini-1.5-flash')
        assert backend.api_key == 'test-key'
        assert backend.model == 'gemini-1.5-flash'
        
        # Test default model
        backend = GeminiVisionBackend(api_key='test-key')
        assert backend.model == 'gemini-1.5-flash'
        
    except ImportError:
        pytest.skip("google-generativeai not installed")


def test_get_vision_backend_with_gemini():
    """Test factory creates Gemini backend correctly."""
    backend = get_vision_backend('gemini', api_key='test-key', model='gemini-1.5-pro')
    
    try:
        import google.generativeai as genai
        from src.vision_backends import GeminiVisionBackend
        
        # Should return Gemini backend with API key
        assert isinstance(backend, GeminiVisionBackend)
        assert backend.api_key == 'test-key'
        assert backend.model == 'gemini-1.5-pro'
        
    except ImportError:
        # Without google-generativeai, should fall back to simulated
        assert isinstance(backend, SimulatedVisionBackend)


def test_get_vision_backend_gemini_without_key():
    """Test that Gemini backend falls back to simulated without API key."""
    backend = get_vision_backend('gemini', api_key=None)
    assert isinstance(backend, SimulatedVisionBackend)
