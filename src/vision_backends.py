"""Vision-capable LLM backends for analyzing video frames."""
import time
import json
import tempfile
import shutil
from typing import Dict, Any, List
from pathlib import Path


class VisionBackendMixin:
    """Mixin class providing video frame analysis capabilities."""
    
    def analyze_video_frames(self, video_path: str, instructions: str, sport: str = "floorball", progress_callback=None) -> Dict[str, Any]:
        """Analyze video by extracting frames and using vision model.
        
        Args:
            video_path: Path to video file
            instructions: What events to find (e.g., "Find all goals and penalties")
            sport: Sport type for context
            progress_callback: Optional callback function(message: str) for progress updates
        
        Returns:
            Dictionary with events, meta information
        """
        from src.video_tools import extract_frames, encode_image_base64, get_video_duration
        from src.config_manager import SPORT_PRESETS
        
        start = time.time()
        temp_dir = tempfile.mkdtemp()
        
        # Get sport-specific settings
        sport_preset = SPORT_PRESETS.get(sport, SPORT_PRESETS.get("floorball"))
        frame_interval = sport_preset.frame_interval
        max_frames_per_call = sport_preset.max_frames
        
        try:
            # Extract frames at sport-specific interval
            frames = extract_frames(video_path, temp_dir, interval_seconds=frame_interval)
            
            if not frames:
                return {
                    "events": [],
                    "meta": {
                        "error": "No frames extracted - ffmpeg may not be installed",
                        "processing_ms": int((time.time() - start) * 1000)
                    }
                }
            
            # Get video duration
            duration = get_video_duration(video_path)
            actual_frame_interval = duration / len(frames) if frames and duration > 0 else frame_interval
            
            # For long videos, process in chunks to avoid missing events
            # Each chunk covers max_frames_per_call * frame_interval seconds
            chunk_duration_seconds = max_frames_per_call * frame_interval * 2  # 2x overlap for safety
            needs_chunking = duration > chunk_duration_seconds and len(frames) > max_frames_per_call * 1.5
            
            if needs_chunking:
                import logging
                from concurrent.futures import ThreadPoolExecutor, as_completed
                logger = logging.getLogger('floorball_llm')
                
                chunk_size = max_frames_per_call
                total_chunks = (len(frames) + (chunk_size // 2) - 1) // (chunk_size // 2)
                
                msg = f"Long video detected ({duration:.1f}s). Processing {total_chunks} chunks in parallel..."
                print(msg)
                logger.info(f"Long video detected: {duration:.1f}s, {len(frames)} frames total")
                logger.info(f"Processing in ~{total_chunks} overlapping chunks (chunk_size={max_frames_per_call}, overlap=50%) with parallel execution")
                if progress_callback:
                    progress_callback(msg)
                
                # Prepare all chunk tasks
                chunk_tasks = []
                for i in range(0, len(frames), chunk_size // 2):  # 50% overlap
                    chunk_frames = frames[i:i + chunk_size]
                    if len(chunk_frames) < 3:  # Skip very small chunks at the end
                        continue
                    
                    chunk_start_time = i * actual_frame_interval
                    chunk_end_time = (i + len(chunk_frames)) * actual_frame_interval
                    chunk_num = len(chunk_tasks) + 1
                    
                    chunk_tasks.append({
                        'chunk_num': chunk_num,
                        'chunk_frames': chunk_frames,
                        'chunk_start_time': chunk_start_time,
                        'chunk_end_time': chunk_end_time,
                        'frame_start': i,
                        'frame_end': i + len(chunk_frames) - 1
                    })
                
                # Process chunks in parallel with ThreadPoolExecutor
                all_events = []
                completed = 0
                max_workers = min(4, len(chunk_tasks))  # Max 4 parallel API calls
                
                def process_chunk(task):
                    """Process a single chunk and return events."""
                    chunk_events = self._analyze_frames_impl(
                        frames=task['chunk_frames'],
                        instructions=instructions,
                        sport=sport,
                        frame_interval=actual_frame_interval,
                        max_frames=len(task['chunk_frames']),
                        time_offset=task['chunk_start_time']
                    )
                    return task, chunk_events
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_task = {executor.submit(process_chunk, task): task for task in chunk_tasks}
                    
                    # Process results as they complete
                    for future in as_completed(future_to_task):
                        task, chunk_events = future.result()
                        completed += 1
                        
                        msg = f"Chunk {completed}/{len(chunk_tasks)}: {task['chunk_start_time']:.0f}s-{task['chunk_end_time']:.0f}s → {len(chunk_events)} events"
                        print(f"  ✓ {msg}")
                        logger.info(f"Chunk {task['chunk_num']} complete: Found {len(chunk_events)} events")
                        if progress_callback:
                            progress_callback(msg)
                        
                        all_events.extend(chunk_events)
                
                # Remove duplicate events (from overlapping chunks)
                events = self._deduplicate_events(all_events, tolerance_seconds=5.0)
                msg = f"Parallel processing complete: {len(all_events)} raw events → {len(events)} unique events"
                print(f"  {msg}")
                logger.info(f"All chunks processed: {len(all_events)} raw events -> {len(events)} unique events after deduplication")
                if progress_callback:
                    progress_callback(msg)
            else:
                # Short video: process normally
                events = self._analyze_frames_impl(
                    frames=frames,
                    instructions=instructions,
                    sport=sport,
                    frame_interval=actual_frame_interval,
                    max_frames=max_frames_per_call,
                    time_offset=0.0
                )
            
            processing_ms = int((time.time() - start) * 1000)
            
            return {
                "events": events,
                "meta": {
                    "processing_ms": processing_ms,
                    "frames_analyzed": len(frames),
                    "video_duration": duration,
                    "frame_interval": frame_interval,
                    "instructions": instructions
                }
            }
        
        except Exception as e:
            return {
                "events": [],
                "meta": {
                    "error": str(e),
                    "processing_ms": int((time.time() - start) * 1000)
                }
            }
        finally:
            # Cleanup temp frames
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _deduplicate_events(self, events: List[Dict[str, Any]], tolerance_seconds: float = 5.0) -> List[Dict[str, Any]]:
        """Remove duplicate events that occur within tolerance window.
        
        Args:
            events: List of events with timestamps
            tolerance_seconds: Events within this time window are considered duplicates
        
        Returns:
            Deduplicated list of events
        """
        if not events:
            return []
        
        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', 0))
        unique_events = []
        
        for event in sorted_events:
            # Check if this event is too close to any existing unique event
            is_duplicate = False
            for unique_event in unique_events:
                time_diff = abs(event.get('timestamp', 0) - unique_event.get('timestamp', 0))
                same_type = event.get('type') == unique_event.get('type')
                
                if time_diff < tolerance_seconds and same_type:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if event.get('confidence', 0) > unique_event.get('confidence', 0):
                        unique_events.remove(unique_event)
                        unique_events.append(event)
                    break
            
            if not is_duplicate:
                unique_events.append(event)
        
        return sorted(unique_events, key=lambda e: e.get('timestamp', 0))
    
    def _analyze_frames_impl(self, frames: List[str], instructions: str, sport: str, frame_interval: float, max_frames: int = 20, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        """Override this method in subclasses to implement specific vision model logic.
        
        Args:
            frames: List of frame file paths
            instructions: What to look for in the video
            sport: Sport being analyzed
            frame_interval: Seconds between frames
            max_frames: Maximum frames to send in single API call
            time_offset: Time offset to add to all timestamps (for chunked processing)
        """
        raise NotImplementedError("Subclasses must implement _analyze_frames_impl")
    
    @staticmethod
    def parse_events_from_text(text: str) -> List[Dict[str, Any]]:
        """Parse event JSON objects from LLM response text."""
        import re
        events = []
        
        # Find JSON objects in text
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                event = json.loads(match)
                # Validate required fields
                if 'timestamp' in event and 'type' in event:
                    # Ensure proper types
                    event['timestamp'] = float(event['timestamp'])
                    event['confidence'] = float(event.get('confidence', 0.8))
                    event['description'] = str(event.get('description', ''))
                    events.append(event)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        
        return events


class OpenAIVisionBackend(VisionBackendMixin):
    """OpenAI GPT-4o Vision backend for video analysis."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, timeout=120.0, max_retries=3)
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def _analyze_frames_impl(self, frames: List[str], instructions: str, sport: str, frame_interval: float, max_frames: int = 20, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        """Analyze frames using GPT-4o Vision."""
        from src.video_tools import encode_image_base64
        
        system_prompt = f"""You are analyzing {sport} game footage. 

User instructions: {instructions}

For each event you identify, respond ONLY with valid JSON objects in this format:
{{"timestamp": <seconds>, "type": "<goal|shot|penalty|save|turnover|assist|timeout>", "description": "<what happened>", "confidence": <0.0-1.0>}}

Provide one JSON object per event, each on a new line."""
        
        # Sample frames based on max_frames setting
        sample_frames = frames[::max(1, len(frames) // max_frames)]
        
        messages = []
        for i, frame_path in enumerate(sample_frames):
            timestamp = i * frame_interval * (len(frames) / len(sample_frames))
            base64_image = encode_image_base64(frame_path)
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Frame at {timestamp:.1f}s:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            })
        
        # Add system instruction at the end
        messages.append({
            "role": "user",
            "content": system_prompt
        })
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=2000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            events = self.parse_events_from_text(content)
            
            # Apply time offset for chunked processing
            if time_offset > 0:
                for event in events:
                    event['timestamp'] += time_offset
            
            return events
        
        except Exception as e:
            print(f"OpenAI Vision error: {e}")
            return []


class SimulatedVisionBackend(VisionBackendMixin):
    """Simulated vision backend for testing without API costs."""
    
    def _analyze_frames_impl(self, frames: List[str], instructions: str, sport: str, frame_interval: float, max_frames: int = 20, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        """Generate simulated events based on instructions."""
        import random
        
        events = []
        instructions_lower = instructions.lower()
        
        # Detect what to find from instructions
        event_types = []
        if 'goal' in instructions_lower:
            event_types.append('goal')
        if 'shot' in instructions_lower:
            event_types.append('shot')
        if 'penalty' in instructions_lower or 'penalt' in instructions_lower:
            event_types.append('penalty')
        if 'save' in instructions_lower:
            event_types.append('save')
        if 'loss' in instructions_lower or 'turnover' in instructions_lower:
            event_types.append('turnover')
        
        if not event_types:
            event_types = ['goal', 'shot', 'penalty']
        
        # Generate events based on frame count
        num_events = min(len(frames) // 3, 10)  # ~1 event per 3 frames, max 10
        
        for i in range(num_events):
            event_type = random.choice(event_types)
            timestamp = (i + 1) * (len(frames) * frame_interval) / (num_events + 1) + time_offset
            
            events.append({
                'type': event_type,
                'timestamp': timestamp,
                'description': f"Simulated {event_type} detected at {timestamp:.1f}s",
                'confidence': round(random.uniform(0.7, 0.95), 2)
            })
        
        return sorted(events, key=lambda x: x['timestamp'])


class GeminiVisionBackend(VisionBackendMixin):
    """Google Gemini Vision backend for video analysis."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        # Normalize model name - remove -latest suffix and models/ prefix if present
        # Gemini SDK wants just the base name like "gemini-1.5-flash"
        model = model.replace('-latest', '').replace('models/', '')
        self.model = model
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai package required: pip install google-generativeai")
    
    def _analyze_frames_impl(self, frames: List[str], instructions: str, sport: str, frame_interval: float, max_frames: int = 20, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        """Analyze frames using Gemini Vision."""
        import google.generativeai as genai
        from PIL import Image
        
        system_prompt = f"""You are analyzing {sport} game footage. 

User instructions: {instructions}

For each event you identify, respond with valid JSON objects in this format:
{{"timestamp": <seconds>, "type": "<goal|shot|penalty|save|turnover|assist|timeout>", "description": "<what happened>", "confidence": <0.0-1.0>}}

Provide one JSON object per event, each on a new line. Only output JSON, no other text."""
        
        # Sample frames based on max_frames setting
        sample_frames = frames[::max(1, len(frames) // max_frames)]
        
        # Prepare content with images
        content = [system_prompt]
        
        for i, frame_path in enumerate(sample_frames):
            timestamp = i * frame_interval * (len(frames) / len(sample_frames))
            
            # Load image using PIL
            try:
                img = Image.open(frame_path)
                content.append(f"\n\nFrame at {timestamp:.1f}s:")
                content.append(img)
            except Exception as e:
                print(f"Warning: Could not load frame {frame_path}: {e}")
                continue
        
        try:
            # Import safety settings enums
            import google.generativeai as genai
            
            # Generate response with proper safety settings
            response = self.client.generate_content(
                content,
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': 2000,
                },
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                safety_ratings = response.candidates[0].safety_ratings if response.candidates else []
                
                error_msg = f"Gemini safety filter blocked the response (finish_reason: {finish_reason})"
                print(f"\n{'='*60}")
                print(f"ERROR: {error_msg}")
                print(f"Safety ratings: {safety_ratings}")
                print(f"\nThis happens with Gemini 3 Pro Preview. SOLUTIONS:")
                print(f"  1. Switch to 'gemini-1.5-flash' model (less restrictive)")
                print(f"  2. Go to Settings → Change gemini_model to 'gemini-1.5-flash'")
                print(f"  3. Or use OpenAI backend for more reliable results")
                print(f"  4. Try shorter video clips (current: {len(frames)} frames)")
                print(f"{'='*60}\n")
                
                # Return error in meta so frontend can show it
                return []
            
            # Parse response
            response_text = response.text
            events = self._parse_events_from_text(response_text)
            
            # Apply time offset for chunked processing
            if time_offset > 0:
                for event in events:
                    event['timestamp'] += time_offset
            
            return events
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            import traceback
            traceback.print_exc()
            return []


def get_vision_backend(backend_name: str, api_key: str = None, model: str = None):
    """Factory function to get vision backend by name."""
    if backend_name == 'openai' and api_key:
        return OpenAIVisionBackend(api_key)
    elif backend_name == 'gemini' and api_key:
        return GeminiVisionBackend(api_key, model or "gemini-1.5-flash")
    elif backend_name == 'simulated' or not api_key:
        return SimulatedVisionBackend()
    else:
        raise ValueError(f"Vision backend '{backend_name}' not supported. Use 'openai', 'gemini', or 'simulated'.")
