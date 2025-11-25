"""Tests for video clipping functionality."""
import pytest
import tempfile
import os
from pathlib import Path
from src.video_clipper import (
    clip_video,
    concatenate_clips,
    get_available_clipping_methods,
    clip_video_ffmpeg_python,
    clip_video_moviepy,
    clip_video_ffmpeg_subprocess
)


class TestAvailableMethods:
    """Test detection of available clipping methods."""
    
    def test_get_available_methods_returns_list(self):
        """Should return a list of available methods."""
        methods = get_available_clipping_methods()
        assert isinstance(methods, list)
    
    def test_at_least_one_method_available(self):
        """Should have at least one clipping method available."""
        methods = get_available_clipping_methods()
        # In test environment, at least ffmpeg-python should be available
        assert len(methods) > 0
    
    def test_methods_are_valid_strings(self):
        """All returned methods should be valid strings."""
        methods = get_available_clipping_methods()
        valid_methods = ['ffmpeg-python', 'ffmpeg-subprocess', 'moviepy']
        for method in methods:
            assert method in valid_methods


class TestClipVideoFallback:
    """Test the fallback logic of clip_video function."""
    
    def test_clip_video_returns_boolean(self):
        """Should return boolean indicating success/failure."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
            output_path = f.name.replace('.mp4', '_clip.mp4')
        
        try:
            # This will fail but should return False, not raise exception
            result = clip_video(video_path, 0, 5, output_path)
            assert isinstance(result, bool)
        finally:
            # Cleanup
            for path in [video_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_clip_video_with_invalid_input_returns_false(self):
        """Should return False with invalid input without crashing."""
        result = clip_video(
            '/nonexistent/video.mp4',
            0,
            5,
            '/tmp/output.mp4'
        )
        assert result is False


class TestConcatenateClipsFallback:
    """Test the fallback logic of concatenate_clips function."""
    
    def test_concatenate_empty_list_returns_false(self):
        """Should return False when given empty list."""
        result = concatenate_clips([], '/tmp/output.mp4')
        assert result is False
    
    def test_concatenate_returns_boolean(self):
        """Should return boolean indicating success/failure."""
        result = concatenate_clips(
            ['/nonexistent1.mp4', '/nonexistent2.mp4'],
            '/tmp/output.mp4'
        )
        assert isinstance(result, bool)


class TestIndividualClippingMethods:
    """Test individual clipping method functions."""
    
    def test_ffmpeg_python_with_invalid_input(self):
        """ffmpeg-python should handle invalid input gracefully."""
        result = clip_video_ffmpeg_python(
            '/nonexistent/video.mp4',
            0,
            5,
            '/tmp/output.mp4'
        )
        assert result is False
    
    def test_moviepy_with_invalid_input(self):
        """moviepy should handle invalid input gracefully."""
        result = clip_video_moviepy(
            '/nonexistent/video.mp4',
            0,
            5,
            '/tmp/output.mp4'
        )
        assert result is False
    
    def test_ffmpeg_subprocess_with_invalid_input(self):
        """ffmpeg subprocess should handle invalid input gracefully."""
        result = clip_video_ffmpeg_subprocess(
            '/nonexistent/video.mp4',
            0,
            5,
            '/tmp/output.mp4'
        )
        assert result is False


class TestVideoToolsIntegration:
    """Test integration with video_tools module."""
    
    def test_extract_clip_uses_clipper(self):
        """extract_clip should use the new clipper module."""
        from src.video_tools import extract_clip
        
        # Should return False with invalid input, not raise exception
        result = extract_clip(
            '/nonexistent/video.mp4',
            0,
            5,
            '/tmp/output.mp4'
        )
        assert result is False
    
    def test_concatenate_clips_uses_clipper(self):
        """concatenate_clips should use the new clipper module."""
        from src.video_tools import concatenate_clips
        
        # Should return False with empty list
        result = concatenate_clips([], '/tmp/output.mp4')
        assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
