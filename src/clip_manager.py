import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from src.schema import Event, EventType


@dataclass
class ClipConfig:
    """Configuration for clip generation."""
    padding_before: int = 5  # seconds
    padding_after: int = 8
    min_duration: int = 3
    max_duration: int = 30
    format: str = "mp4"
    codec: str = "libx264"
    quality: str = "medium"  # low, medium, high


class ClipManager:
    """Manages video clip extraction and compilation."""
    
    def __init__(self, config: ClipConfig = None):
        self.config = config or ClipConfig()
    
    def prepare_clips(self, events: List[Event], video_path: str, out_dir: str) -> List[Dict[str, Any]]:
        """Prepare clip metadata for extraction (placeholder for now)."""
        os.makedirs(out_dir, exist_ok=True)
        clips = []
        
        for i, event in enumerate(events):
            start_time = max(0, event.timestamp - self.config.padding_before)
            end_time = event.timestamp + self.config.padding_after
            duration = end_time - start_time
            
            # Ensure duration constraints
            if duration < self.config.min_duration:
                padding_needed = self.config.min_duration - duration
                end_time += padding_needed
                duration = self.config.min_duration
            
            if duration > self.config.max_duration:
                end_time = start_time + self.config.max_duration
                duration = self.config.max_duration
            
            clip_filename = f"clip_{i:03d}_{event.type}_{int(event.timestamp)}.{self.config.format}"
            clip_path = os.path.join(out_dir, clip_filename)
            
            # Create placeholder file
            with open(clip_path, "wb") as f:
                f.write(b"")
            
            clip_meta = {
                "path": clip_path,
                "event": event.model_dump(),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "index": i
            }
            clips.append(clip_meta)
        
        return clips
    
    def create_highlight_reel(
        self,
        clips: List[Dict[str, Any]],
        output_path: str,
        event_types: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """Create a highlight reel from selected clips."""
        # Filter clips by criteria
        filtered_clips = []
        for clip in clips:
            event = clip["event"]
            if event_types and event["type"] not in event_types:
                continue
            if event.get("confidence", 1.0) < min_confidence:
                continue
            filtered_clips.append(clip)
        
        # Sort by timestamp
        filtered_clips.sort(key=lambda c: c["start_time"])
        
        # Create placeholder highlight reel file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"")
        
        return {
            "output_path": output_path,
            "clips_count": len(filtered_clips),
            "total_duration": sum(c["duration"] for c in filtered_clips),
            "clip_indices": [c["index"] for c in filtered_clips]
        }
    
    def create_player_compilation(
        self,
        clips: List[Dict[str, Any]],
        player_name: str,
        output_path: str
    ) -> Dict[str, Any]:
        """Create a compilation of clips featuring a specific player."""
        player_clips = [
            c for c in clips
            if c["event"].get("player", "").lower() == player_name.lower()
        ]
        
        player_clips.sort(key=lambda c: c["start_time"])
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"")
        
        return {
            "output_path": output_path,
            "player": player_name,
            "clips_count": len(player_clips),
            "total_duration": sum(c["duration"] for c in player_clips),
            "event_breakdown": self._event_breakdown(player_clips)
        }
    
    def create_team_compilation(
        self,
        clips: List[Dict[str, Any]],
        team_name: str,
        output_path: str
    ) -> Dict[str, Any]:
        """Create a compilation of clips featuring a specific team."""
        team_clips = [
            c for c in clips
            if c["event"].get("team", "").lower() == team_name.lower()
        ]
        
        team_clips.sort(key=lambda c: c["start_time"])
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"")
        
        return {
            "output_path": output_path,
            "team": team_name,
            "clips_count": len(team_clips),
            "total_duration": sum(c["duration"] for c in team_clips),
            "event_breakdown": self._event_breakdown(team_clips)
        }
    
    def _event_breakdown(self, clips: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count events by type."""
        breakdown = {}
        for clip in clips:
            event_type = clip["event"]["type"]
            breakdown[event_type] = breakdown.get(event_type, 0) + 1
        return breakdown
    
    def get_clip_config_for_event(self, event_type: str) -> ClipConfig:
        """Get optimized clip configuration for specific event type."""
        # Customize padding based on event type
        configs = {
            "goal": ClipConfig(padding_before=8, padding_after=12),
            "penalty": ClipConfig(padding_before=10, padding_after=5),
            "shot": ClipConfig(padding_before=3, padding_after=5),
            "save": ClipConfig(padding_before=3, padding_after=5),
            "timeout": ClipConfig(padding_before=2, padding_after=2),
        }
        return configs.get(event_type, self.config)
