"""Video clipping utilities with multiple backend options."""
import os
import subprocess
from typing import List, Optional
from pathlib import Path

def clip_video_ffmpeg_python(video_path: str, start_time: float, duration: float, output_path: str) -> bool:
    """Clip video using ffmpeg-python library (requires ffmpeg installed).
    
    Args:
        video_path: Source video path
        start_time: Start time in seconds
        duration: Clip duration in seconds
        output_path: Output clip path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import ffmpeg
        
        (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .output(output_path, c='copy', loglevel='error')
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except (ImportError, subprocess.CalledProcessError, Exception) as e:
        print(f"ffmpeg-python failed: {e}")
        return False


def clip_video_moviepy(video_path: str, start_time: float, duration: float, output_path: str) -> bool:
    """Clip video using moviepy library (pure Python, no ffmpeg needed).
    
    Args:
        video_path: Source video path
        start_time: Start time in seconds
        duration: Clip duration in seconds
        output_path: Output clip path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        
        clip = VideoFileClip(video_path).subclipped(start_time, start_time + duration)
        clip.write_videofile(output_path, logger=None)
        clip.close()
        return True
    except (ImportError, Exception) as e:
        print(f"moviepy failed: {e}")
        return False


def clip_video_ffmpeg_subprocess(video_path: str, start_time: float, duration: float, output_path: str) -> bool:
    """Clip video using ffmpeg subprocess (requires ffmpeg installed).
    
    Args:
        video_path: Source video path
        start_time: Start time in seconds
        duration: Clip duration in seconds
        output_path: Output clip path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-c', 'copy',  # Fast copy without re-encoding
            '-y',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"ffmpeg subprocess failed: {e}")
        return False


def clip_video(video_path: str, start_time: float, duration: float, output_path: str) -> bool:
    """Clip video using best available method (tries multiple backends).
    
    Priority order:
    1. ffmpeg-python (fast, requires ffmpeg)
    2. ffmpeg subprocess (fast, requires ffmpeg) 
    3. moviepy (slower but works without ffmpeg)
    
    Args:
        video_path: Source video path
        start_time: Start time in seconds
        duration: Clip duration in seconds
        output_path: Output clip path
    
    Returns:
        True if successful, False if all methods failed
    """
    # Try ffmpeg-python first (fastest with copy mode)
    if clip_video_ffmpeg_python(video_path, start_time, duration, output_path):
        return True
    
    # Try ffmpeg subprocess (also fast)
    if clip_video_ffmpeg_subprocess(video_path, start_time, duration, output_path):
        return True
    
    # Fallback to moviepy (slower but doesn't require ffmpeg)
    if clip_video_moviepy(video_path, start_time, duration, output_path):
        return True
    
    return False


def concatenate_clips_ffmpeg_python(clip_paths: List[str], output_path: str) -> bool:
    """Concatenate clips using ffmpeg-python.
    
    Args:
        clip_paths: List of clip file paths
        output_path: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import ffmpeg
        import tempfile
        
        # Create concat file list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            list_file = f.name
            for clip in clip_paths:
                f.write(f"file '{os.path.abspath(clip)}'\n")
        
        try:
            (
                ffmpeg
                .input(list_file, format='concat', safe=0)
                .output(output_path, c='copy', loglevel='error')
                .overwrite_output()
                .run(quiet=True)
            )
            return True
        finally:
            os.unlink(list_file)
            
    except (ImportError, subprocess.CalledProcessError, Exception) as e:
        print(f"ffmpeg-python concatenation failed: {e}")
        return False


def concatenate_clips_moviepy(clip_paths: List[str], output_path: str) -> bool:
    """Concatenate clips using moviepy.
    
    Args:
        clip_paths: List of clip file paths
        output_path: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy import concatenate_videoclips
        
        clips = [VideoFileClip(path) for path in clip_paths]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, logger=None)
        
        # Cleanup
        for clip in clips:
            clip.close()
        final_clip.close()
        
        return True
    except (ImportError, Exception) as e:
        print(f"moviepy concatenation failed: {e}")
        return False


def concatenate_clips(clip_paths: List[str], output_path: str) -> bool:
    """Concatenate clips using best available method.
    
    Args:
        clip_paths: List of clip file paths
        output_path: Output file path
    
    Returns:
        True if successful, False if all methods failed
    """
    if not clip_paths:
        return False
    
    # Try ffmpeg-python first (fastest)
    if concatenate_clips_ffmpeg_python(clip_paths, output_path):
        return True
    
    # Fallback to moviepy
    if concatenate_clips_moviepy(clip_paths, output_path):
        return True
    
    return False


def get_available_clipping_methods() -> List[str]:
    """Check which clipping methods are available.
    
    Returns:
        List of available method names
    """
    methods = []
    
    # Check ffmpeg-python
    try:
        import ffmpeg  # type: ignore
        methods.append('ffmpeg-python')
    except ImportError:
        pass
    
    # Check ffmpeg subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
        methods.append('ffmpeg-subprocess')
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check moviepy
    try:
        import moviepy  # type: ignore
        methods.append('moviepy')
    except ImportError:
        pass
    
    return methods
