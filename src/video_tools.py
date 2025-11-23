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


def extract_frames(video_path: str, output_dir: str, interval_seconds: int = 5) -> List[str]:
    """Extract frames from video at specified intervals.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        interval_seconds: Extract one frame every N seconds
    
    Returns:
        List of frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use ffmpeg to extract frames
        output_pattern = os.path.join(output_dir, 'frame_%04d.jpg')
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps=1/{interval_seconds}',  # 1 frame every N seconds
            '-q:v', '2',  # High quality
            output_pattern,
            '-y'  # Overwrite
        ]
        
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
    """Extract a clip from video using ffmpeg.
    
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
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def prepare_clips(events: List[Dict[str, Any]], video_path: str, out_dir: str, padding: int = 5) -> List[str]:
    """Extract video clips for detected events.
    
    Args:
        events: List of event dictionaries with timestamp
        video_path: Source video path
        out_dir: Output directory for clips
        padding: Seconds to include before/after event
    
    Returns:
        List of generated clip paths
    """
    os.makedirs(out_dir, exist_ok=True)
    clips = []
    
    for i, ev in enumerate(events):
        timestamp = ev.get("timestamp", ev.get("timestamp_seconds", 0))
        start = max(0, timestamp - padding)
        duration = padding * 2
        
        event_type = ev.get('type', 'event')
        clip_path = os.path.join(out_dir, f"clip_{i:03d}_{event_type}_{int(timestamp)}.mp4")
        
        if extract_clip(video_path, start, duration, clip_path):
            clips.append(clip_path)
    
    return clips


def concatenate_clips(clip_paths: List[str], output_path: str) -> bool:
    """Concatenate multiple video clips into one video.
    
    Args:
        clip_paths: List of clip file paths to concatenate
        output_path: Output path for concatenated video
    
    Returns:
        True if successful, False otherwise
    """
    if not clip_paths:
        return False
    
    import tempfile
    
    try:
        # Create a temporary file list for ffmpeg concat
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            list_file = f.name
            for clip in clip_paths:
                # Write in ffmpeg concat format
                f.write(f"file '{os.path.abspath(clip)}'\n")
        
        # Use ffmpeg to concatenate
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',  # Fast copy without re-encoding
            '-y',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Cleanup temp file
        os.unlink(list_file)
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error concatenating clips: {e}")
        return False

