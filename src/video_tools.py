import os
import subprocess
import json
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0.0


def extract_frames(video_path: str, output_dir: str, interval_seconds: float = 5.0, start_time: float = None, end_time: float = None) -> List[str]:
    """Extract frames from video at specified intervals.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        interval_seconds: Extract one frame every N seconds (can be fractional)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
    
    Returns:
        List of frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Build ffmpeg command
        cmd = ['ffmpeg']
        
        # Add start time if specified
        if start_time is not None and start_time > 0:
            cmd.extend(['-ss', str(start_time)])
        
        # Add input file
        cmd.extend(['-i', video_path])
        
        # Add duration if end_time is specified
        if end_time is not None and start_time is not None:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])
        elif end_time is not None:
            cmd.extend(['-t', str(end_time)])
        
        # Add frame extraction filter and output
        output_pattern = os.path.join(output_dir, 'frame_%04d.jpg')
        # Protect against zero or negative intervals
        safe_interval = max(0.05, interval_seconds)
        cmd.extend([
            '-vf', f'fps=1/{safe_interval}',  # 1 frame every N seconds
            '-q:v', '2',  # High quality
            output_pattern,
            '-y'  # Overwrite
        ])
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Get list of generated frames
        frames = sorted([str(f) for f in Path(output_dir).glob('frame_*.jpg')])
        return frames
    
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Frame extraction failed: {e}")
        return []


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for LLM API."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def find_audio_transcript_for_video(video_path: str) -> str:
    """Extract or locate an existing transcript for the video.
    For the prototype we'll just look for a .txt with same base name.
    """
    base, _ = os.path.splitext(video_path)
    txt = base + ".txt"
    if os.path.exists(txt):
        with open(txt, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def extract_clip(video_path: str, start_time: float, duration: float, output_path: str) -> bool:
    """Extract a clip from video using multiple methods (ffmpeg-python, moviepy fallback).
    
    Args:
        video_path: Source video path
        start_time: Start time in seconds
        duration: Clip duration in seconds
        output_path: Output clip path
    
    Returns:
        True if successful, False otherwise
    """
    from src.video_clipper import clip_video
    return clip_video(video_path, start_time, duration, output_path)


def prepare_clips(events: List[Dict[str, Any]], video_path: str, out_dir: str, padding_before: int = 5, padding_after: int = 5) -> List[str]:
    """Extract video clips for detected events.
    
    Args:
        events: List of event dictionaries with timestamp
        video_path: Source video path
        out_dir: Output directory for clips
        padding_before: Seconds to include before event
        padding_after: Seconds to include after event
    
    Returns:
        List of generated clip paths
    """
    os.makedirs(out_dir, exist_ok=True)
    clips = []
    
    for i, ev in enumerate(events):
        timestamp = ev.get("timestamp", ev.get("timestamp_seconds", 0))
        start = max(0, timestamp - padding_before)
        duration = padding_before + padding_after
        
        event_type = ev.get('type', 'event')
        clip_path = os.path.join(out_dir, f"clip_{i:03d}_{event_type}_{int(timestamp)}.mp4")
        
        if extract_clip(video_path, start, duration, clip_path):
            clips.append(clip_path)
    
    return clips


def concatenate_clips(clip_paths: List[str], output_path: str) -> bool:
    """Concatenate multiple video clips into one video using multiple methods.
    
    Args:
        clip_paths: List of clip file paths to concatenate
        output_path: Output path for concatenated video
    
    Returns:
        True if successful, False otherwise
    """
    from src.video_clipper import concatenate_clips as concat_clips
    return concat_clips(clip_paths, output_path)

