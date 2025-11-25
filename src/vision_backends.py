"""Vision-capable LLM backends for analyzing video frames."""
import os
import time
import json
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path


class VisionBackendMixin:
    """Mixin class providing video frame analysis capabilities."""
    
    def analyze_video_frames(self, video_path: str, instructions: str, sport: str = "floorball", progress_callback: Optional[Callable[[str], None]] = None, config: Optional[Any] = None, is_cancelled_callback: Optional[Callable[[], bool]] = None, time_from: Optional[float] = None, time_to: Optional[float] = None) -> Dict[str, Any]:
        """Analyze video by extracting frames and using vision model.
        
        Args:
            video_path: Path to video file
            instructions: What events to find (e.g., "Find all goals and penalties")
            sport: Sport type for context
            progress_callback: Optional callback function(message: str) for progress updates
            config: Optional AppConfig object with rate limit settings
            is_cancelled_callback: Optional callback function() -> bool to check if analysis should stop
            time_from: Optional start time in seconds
            time_to: Optional end time in seconds
        
        Returns:
            Dictionary with events, meta information
        """
        # Get rate limit settings from config or use defaults
        max_workers_openai = None
        max_workers_gemini = 4
        rate_limit_retry_delay = 40.0
        rate_limit_max_retries = 3
        
        if config:
            # Check if we should calculate workers automatically based on rate limits
            openai_tpm = getattr(config, 'openai_rate_limit_tpm', 200000)
            openai_rpm = getattr(config, 'openai_rate_limit_rpm', 500)
            
            # Calculate optimal workers: use RPM as the limiting factor
            # Each chunk takes ~2-5 seconds, so we can process ~12-30 chunks/min per worker
            # Conservative estimate: 15 chunks per minute per worker
            # max_workers = RPM / 15 (rounded down for safety margin)
            # Use 80% of calculated capacity to leave headroom for rate limit variations
            calculated_openai_workers = max(1, int((openai_rpm // 15) * 0.8))
            
            # Allow manual override if explicitly set, otherwise use calculated
            configured_workers = getattr(config, 'max_workers_openai', None)
            max_workers_openai = configured_workers if configured_workers is not None else calculated_openai_workers
            max_workers_gemini = getattr(config, 'max_workers_gemini', 4)
            rate_limit_retry_delay = getattr(config, 'rate_limit_retry_delay', 40.0)
            rate_limit_max_retries = getattr(config, 'rate_limit_max_retries', 3)
        
        if max_workers_openai is None:
            max_workers_openai = 1

        from src.video_tools import extract_frames, encode_image_base64, get_video_duration
        from src.config_manager import SPORT_PRESETS
        
        start = time.time()
        temp_dir = tempfile.mkdtemp()
        
        # Get sport-specific settings
        sport_preset = SPORT_PRESETS.get(sport, SPORT_PRESETS.get("floorball"))
        if sport_preset is None:
            sport_preset = SPORT_PRESETS["floorball"]  # Fallback to floorball
        frame_interval = sport_preset.frame_interval
        max_frames_per_call = sport_preset.max_frames
        
        # AUTO-CALCULATE max_frames based on TPM/RPM limits if config is provided
        if config:
            # Estimate: Each frame ≈ 850 tokens for vision models (varies by resolution)
            # System prompt ≈ 1000 tokens
            # Response ≈ 500 tokens per event (assume avg 3 events per call)
            TOKENS_PER_FRAME = 850
            SYSTEM_PROMPT_TOKENS = 1000
            RESPONSE_TOKENS = 1500  # Conservative estimate
            
            # Get rate limits
            openai_tpm = getattr(config, 'openai_rate_limit_tpm', 500000)
            openai_rpm = getattr(config, 'openai_rate_limit_rpm', 500)
            
            # Calculate max frames based on TPM limit (most restrictive)
            # max_tokens_per_call = TPM / (60 / call_duration) ≈ TPM / 12 (assume 5s per call)
            # Available for frames = max_tokens_per_call - system_prompt - response
            max_tokens_per_call = openai_tpm / 12  # Conservative: assume 5s per call
            available_tokens = max_tokens_per_call - SYSTEM_PROMPT_TOKENS - RESPONSE_TOKENS
            max_frames_from_tpm = max(5, int(available_tokens / TOKENS_PER_FRAME))
            
            # Use the configured max_frames as upper limit, auto-calculated as optimization
            # If configured value would exceed rate limits, reduce it automatically
            if max_frames_per_call > max_frames_from_tpm:
                import logging
                logger = logging.getLogger('floorball_llm')
                logger.info(f"Auto-adjusting max_frames: {max_frames_per_call} → {max_frames_from_tpm} (based on TPM limit of {openai_tpm})")
                max_frames_per_call = max_frames_from_tpm
        
        try:
            # Extract frames at sport-specific interval
            frames = extract_frames(
                video_path, 
                temp_dir, 
                interval_seconds=frame_interval, 
                start_time=time_from if time_from is not None else 0.0, 
                end_time=time_to if time_to is not None else 0.0
            )
            
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

            def build_meta(events_list: Optional[List[Dict[str, Any]]] = None, *, cancelled: bool = False, chunks_completed: int = 0, total_chunks: int = 0) -> Dict[str, Any]:
                payload_events = events_list if events_list is not None else []
                return {
                    "processing_ms": int((time.time() - start) * 1000),
                    "frames_analyzed": len(frames),
                    "video_duration": duration,
                    "frame_interval": frame_interval,
                    "instructions": instructions,
                    "cancelled": cancelled,
                    "chunks_completed": chunks_completed,
                    "total_chunks": total_chunks,
                    "raw_events": len(payload_events)
                }

            if is_cancelled_callback and is_cancelled_callback():
                return {
                    "events": [],
                    "meta": build_meta(cancelled=True)
                }
            
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
                # Use configured max_workers based on backend type
                is_openai = hasattr(self, 'client') and hasattr(self, 'api_key') and 'openai' in self.__class__.__name__.lower()
                max_workers = max_workers_openai if is_openai else min(max_workers_gemini, len(chunk_tasks))
                
                def process_chunk(task):
                    """Process a single chunk and return events with retry on rate limit."""
                    import time
                    
                    if is_cancelled_callback and is_cancelled_callback():
                        logger.info(f"Cancelling chunk {task['chunk_num']} before request")
                        return task, []

                    for attempt in range(rate_limit_max_retries):
                        try:
                            chunk_events = self._analyze_frames_impl(
                                frames=task['chunk_frames'],
                                instructions=instructions,
                                sport=sport,
                                frame_interval=actual_frame_interval,
                                max_frames=len(task['chunk_frames']),
                                time_offset=task['chunk_start_time']
                            )
                            return task, chunk_events
                        except Exception as e:
                            error_str = str(e)
                            # Check if it's a rate limit error
                            if '429' in error_str or 'rate_limit' in error_str.lower():
                                if attempt < rate_limit_max_retries - 1:
                                    print(f"  ⏳ Rate limit hit on chunk {task['chunk_num']}, waiting {rate_limit_retry_delay}s...")
                                    time.sleep(rate_limit_retry_delay)
                                    continue
                            # Re-raise if not rate limit or last attempt
                            raise
                    
                    return task, []  # Return empty if all retries failed
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_task = {executor.submit(process_chunk, task): task for task in chunk_tasks}
                    
                    # Process results as they complete
                    for future in as_completed(future_to_task):
                        # Check for cancellation
                        if is_cancelled_callback and is_cancelled_callback():
                            logger.info(f"Cancellation detected. Processed {completed}/{len(chunk_tasks)} chunks")
                            print(f"  ⛔ Analysis stopped by user. Processed {completed}/{len(chunk_tasks)} chunks")
                            if progress_callback:
                                progress_callback(f"Stopped - Processed {completed}/{len(chunk_tasks)} chunks")
                            
                            # Cancel remaining futures
                            for f in future_to_task:
                                if not f.done():
                                    f.cancel()
                            
                            # Process events we have so far
                            events = self._deduplicate_events(all_events, tolerance_seconds=5.0)
                            return {
                                "events": events,
                                "meta": build_meta(events, cancelled=True, chunks_completed=completed, total_chunks=len(chunk_tasks))
                            }
                        
                        try:
                            task, chunk_events = future.result()
                            completed += 1
                            
                            msg = f"Chunk {completed}/{len(chunk_tasks)}: {task['chunk_start_time']:.0f}s-{task['chunk_end_time']:.0f}s → {len(chunk_events)} events"
                            print(f"  ✓ {msg}")
                            logger.info(f"Chunk {task['chunk_num']} complete: Found {len(chunk_events)} events")
                            if progress_callback:
                                progress_callback(msg)
                            
                            all_events.extend(chunk_events)
                        except Exception as e:
                            # Log error but continue with other chunks
                            print(f"  ✗ Chunk {future_to_task[future]['chunk_num']} failed: {str(e)[:100]}")
                            logger.error(f"Chunk {future_to_task[future]['chunk_num']} failed: {e}")
                            completed += 1
                
                # Remove duplicate events (from overlapping chunks)
                events = self._deduplicate_events(all_events, tolerance_seconds=5.0)
                msg = f"Parallel processing complete: {len(all_events)} raw events → {len(events)} unique events"
                print(f"  {msg}")
                logger.info(f"All chunks processed: {len(all_events)} raw events -> {len(events)} unique events after deduplication")
                if progress_callback:
                    progress_callback(msg)
                if is_cancelled_callback and is_cancelled_callback():
                    return {
                        "events": events,
                        "meta": build_meta(events, cancelled=True, chunks_completed=len(chunk_tasks), total_chunks=len(chunk_tasks))
                    }
                events = self._postprocess_events(events, video_path, instructions, sport, max(actual_frame_interval, frame_interval), config, progress_callback)
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
                if is_cancelled_callback and is_cancelled_callback():
                    return {
                        "events": events,
                        "meta": build_meta(events, cancelled=True)
                    }
                events = self._postprocess_events(events, video_path, instructions, sport, max(actual_frame_interval, frame_interval), config, progress_callback)
            
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

    def _postprocess_events(self, events: List[Dict[str, Any]], video_path: str, instructions: str, sport: str, frame_interval: float, config: Optional[Any], progress_callback: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
        """Apply confirmation, refinement, and annotation logic to goal events."""
        if not events:
            return []

        events = self._confirm_goal_events(events)
        events = self._refine_goal_candidates(events, video_path, instructions, sport, frame_interval, config, progress_callback)
        self._persist_goal_annotations(events, video_path, config)
        return events

    def _confirm_goal_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Boost or lower goal confidence based on supporting events"""
        updated_events = []
        for event in events:
            event_copy = dict(event)
            if event_copy.get('type') == 'goal':
                description = event_copy.get('description', '').lower()
                scoreboard_only = any(token in description for token in ['scoreboard', 'score board', 'score update', 'score change'])
                supporting_types = {'shot', 'save'}
                supporting = any(
                    abs(event_copy.get('timestamp', 0) - neighbor.get('timestamp', 0)) <= 3 and neighbor.get('type') in supporting_types
                    for neighbor in events if neighbor is not event
                )

                if scoreboard_only and not supporting:
                    event_copy['confidence'] = min(event_copy.get('confidence', 0.0), 0.65)
                    event_copy['confirmation'] = 'scoreboard_only'
                elif supporting:
                    event_copy['confidence'] = min(1.0, max(event_copy.get('confidence', 0.0), 0.8))
                    event_copy['confirmation'] = 'support_event'
                else:
                    event_copy['confirmation'] = 'auto'

            updated_events.append(event_copy)
        return updated_events

    def _refine_goal_candidates(self, events: List[Dict[str, Any]], video_path: str, instructions: str, sport: str, frame_interval: float, config: Optional[Any], progress_callback: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
        """Run a dense-sampling pass around uncertain goal events to confirm them."""
        refinement_enabled = getattr(config, 'goal_refinement_enabled', True)
        if not refinement_enabled:
            return events

        max_attempts = getattr(config, 'goal_refinement_attempts', 2)
        window = getattr(config, 'goal_refinement_window', 2.5)
        dense_interval = getattr(config, 'goal_refinement_interval', 0.25)

        candidates = [ev for ev in events if ev.get('type') == 'goal' and ev.get('confirmation') != 'support_event']
        if not candidates:
            return events

        from src.video_tools import extract_frames

        for candidate in candidates[:max_attempts]:
            start = max(0.0, candidate.get('timestamp', 0) - window)
            end = candidate.get('timestamp', 0) + window
            with tempfile.TemporaryDirectory() as refine_dir:
                dense_frames = extract_frames(
                    video_path,
                    refine_dir,
                    interval_seconds=dense_interval,
                    start_time=start,
                    end_time=end
                )

                if not dense_frames:
                    continue

                if progress_callback:
                    progress_callback(f"Refining candidate around {candidate.get('timestamp', 0):.1f}s with dense sampling...")

                refinement_instruction = (
                    f"Confirm the potential goal around {candidate.get('timestamp', 0):.1f}s with clear evidence of the ball crossing the line. "
                    "Only report the goal again if you see it."
                )

                refined_events = self._analyze_frames_impl(
                    frames=dense_frames,
                    instructions=refinement_instruction,
                    sport=sport,
                    frame_interval=dense_interval,
                    max_frames=len(dense_frames),
                    time_offset=start
                )

                match = next(
                    (
                        ev for ev in refined_events
                        if ev.get('type') == 'goal' and abs(ev.get('timestamp', 0) - candidate.get('timestamp', 0)) <= 1.5
                    ),
                    None
                )

                if match:
                    candidate['confidence'] = max(candidate.get('confidence', 0.0), match.get('confidence', 0.0))
                    candidate['confirmation'] = 'dense_sampling'

        return events

    def _persist_goal_annotations(self, events: List[Dict[str, Any]], video_path: str, config: Optional[Any]) -> None:
        """Append goal events to annotation log when enabled."""
        enabled = getattr(config, 'goal_annotation_enabled', False)
        if not enabled:
            return

        threshold = getattr(config, 'goal_annotation_threshold', 0.7)
        annotation_dir = Path(getattr(config, 'goal_annotation_dir', 'annotations/goals'))
        annotation_dir.mkdir(parents=True, exist_ok=True)
        annotation_file = annotation_dir / 'goal_candidates.jsonl'

        with annotation_file.open('a', encoding='utf-8') as fh:
            for event in events:
                if event.get('type') != 'goal':
                    continue
                if event.get('confidence', 0.0) < threshold:
                    continue

                record = {
                    'video': os.path.basename(video_path),
                    'timestamp': event.get('timestamp', 0),
                    'confidence': event.get('confidence', 0.0),
                    'description': event.get('description', ''),
                    'confirmation': event.get('confirmation')
                }
                json.dump(record, fh)
                fh.write('\n')
    
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
    """OpenAI Vision backend for video analysis."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, timeout=120.0, max_retries=3)
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def _analyze_frames_impl(self, frames: List[str], instructions: str, sport: str, frame_interval: float, max_frames: int = 20, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        """Analyze frames using GPT-4o Vision.
        
        Args:
            frames: List of file paths OR base64-encoded strings
        """
        from src.video_tools import encode_image_base64
        from src.config_manager import SPORT_PRESETS
        import os
        
        # Get sport-specific hint from config
        sport_preset = SPORT_PRESETS.get(sport) or SPORT_PRESETS.get('floorball')
        if sport_preset is None:
            raise ValueError(f"Sport preset not found for {sport}")
        hint = getattr(sport_preset, 'hint', f"Analyze this {sport} game footage.")
        
        system_prompt = f"""You are an expert {sport} video analyst with years of experience identifying key game events.

## Your Task
Analyze the provided video frames to find ONLY: {instructions}

IMPORTANT: Only detect and report the specific event types mentioned in the instructions above. Do not report other event types.

## Critical Visual Cues for {sport.title()}
{hint}

## Detailed Event Recognition Guidelines

When analyzing frames, look for these SPECIFIC visual indicators:

### GOALS - Highest Priority Visual Evidence:
1. **Ball crossing goal line** - Ball visibly inside or past the goal frame
2. **Net movement** - Goal net shakes, moves, or deforms from ball impact
3. **Goalkeeper position** - Goalkeeper on ground, diving away, or clearly beaten
4. **Player reactions** - Arms raised in celebration, running toward teammates, jumping
5. **Team celebration** - Multiple players hugging, high-fiving, grouping together
6. **Referee signal** - Referee pointing to center circle or making goal signal

### SHOTS - Clear Striking Action:
1. **Wind-up motion** - Player pulls stick back before striking
2. **Strike impact** - Visible moment of stick hitting ball with force
3. **Ball trajectory** - Ball moving rapidly toward goal
4. **Goalkeeper reaction** - Goalkeeper moving, diving, or preparing to block
5. **Shot location** - Player positioned to shoot (not just passing)

### SAVES - Goalkeeper Intervention:
1. **Ball contact** - Goalkeeper's hands, stick, or body clearly touching/blocking ball
2. **Deflection** - Ball direction changes after goalkeeper contact
3. **Ball secured** - Goalkeeper catching or controlling the ball
4. **Diving/stretching** - Goalkeeper extending body to reach the ball
5. **Immediate aftermath** - Ball goes wide, over goal, or out of play after save

### PENALTIES - Official Actions:
1. **Referee arm raised** - Referee's arm up signaling penalty
2. **Player pointing** - Referee pointing at specific player
3. **Play stoppage** - Whistle blown, players stopping
4. **Player leaving** - Penalized player skating to penalty box/bench
5. **Time gesture** - Referee showing 2-minute or 5-minute hand signal

❌ DO NOT REPORT:
- Event types NOT mentioned in the instructions
- Normal passes or ball possession without shooting motion
- Players casually skating or positioning
- Scoreboard changes (these happen after the actual event)
- Ambiguous actions where you cannot clearly identify the event type
- Celebrations without seeing the actual goal being scored

## Output Format (Critical - Follow Exactly)

For EACH event you identify, output valid JSON on a single line:
{{"timestamp": 125.5, "type": "goal", "description": "Player #7 shoots from center, ball enters top right corner of goal, goalkeeper dives but misses, teammates celebrate", "confidence": 0.95}}

**Valid event types**: goal, shot, save, penalty, assist, timeout, turnover

## Boundaries & Confidence Levels
- ✅ ALWAYS: Only report event types explicitly requested in the instructions
- ✅ ALWAYS: Provide specific visual evidence in description (player numbers, jersey colors, positions, actions)
- ✅ ALWAYS: Include timestamp based on frame time
- ✅ HIGH CONFIDENCE (0.85-1.0): Multiple clear visual indicators present (e.g., ball in goal + net moving + celebration)
- ⚠️ MEDIUM CONFIDENCE (0.7-0.85): Most indicators present but some uncertainty (e.g., shot clearly taken but goalkeeper reaction unclear)
- ⚠️ LOW CONFIDENCE (0.5-0.7): Limited visual evidence, use ONLY if event seems likely
- 🚫 NEVER: Output explanatory text, only JSON objects
- 🚫 NEVER: Report events without clear visual evidence
- 🚫 NEVER: Report event types not mentioned in the instructions
- 🚫 NEVER: Guess or infer events from scoreboard changes alone

## Example High-Quality Output
{{"timestamp": 37.5, "type": "goal", "description": "Player in white jersey #10 shoots from 3 meters, ball crosses goal line into bottom left corner, net shakes, goalkeeper on ground, three players raise arms and run together celebrating", "confidence": 0.95}}
{{"timestamp": 42.0, "type": "save", "description": "Goalkeeper in red #1 dives left, catches high shot with both gloves clearly visible, ball secured, shooter in blue stops", "confidence": 0.92}}
{{"timestamp": 89.3, "type": "shot", "description": "Player in blue #15 winds up and strikes ball from right side toward goal, goalkeeper moves to block, ball goes wide right", "confidence": 0.88}}

Analyze ALL frames carefully. Report EVERY event you see that matches the requested types with clear evidence."""
        
        # Sample frames based on max_frames setting
        sample_frames = frames[::max(1, len(frames) // max_frames)]
        
        messages = []
        for i, frame_data in enumerate(sample_frames):
            timestamp = i * frame_interval * (len(frames) / len(sample_frames))
            
            # Check if frame_data is a file path or base64 string
            if os.path.exists(frame_data):
                # It's a file path, encode it
                base64_image = encode_image_base64(frame_data)
            else:
                # It's already base64-encoded
                base64_image = frame_data
            
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
            import logging
            logger = logging.getLogger('floorball_llm')
            logger.info(f"OpenAI Vision: Analyzing {len(sample_frames)} frames (sampled from {len(frames)} total) with model {self.model}")
            
            # Use max_completion_tokens for newer models, max_tokens for older ones
            token_param = {}
            if self.model.startswith('gpt-4o-mini') or self.model.startswith('gpt-5'):
                token_param['max_completion_tokens'] = 4000
            else:
                token_param['max_tokens'] = 4000
            
            # GPT-5 models only support temperature=1 (default)
            api_params = {
                'model': self.model,
                'messages': messages,
                **token_param
            }
            if not self.model.startswith('gpt-5'):
                api_params['temperature'] = 0.3
            
            response = self.client.chat.completions.create(**api_params)
            
            content = response.choices[0].message.content
            logger.info(f"OpenAI Vision raw response length: {len(content)} chars")
            logger.debug(f"OpenAI Vision response: {content[:500]}...")
            
            events = self.parse_events_from_text(content)
            logger.info(f"OpenAI Vision: Parsed {len(events)} events from response")
            
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
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)  # type: ignore
            self.client = genai.GenerativeModel(model)  # type: ignore
        except ImportError:
            raise ImportError("google-generativeai package required: pip install google-generativeai")
    
    def _analyze_frames_impl(self, frames: List[str], instructions: str, sport: str, frame_interval: float, max_frames: int = 20, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        """Analyze frames using Gemini Vision."""
        import google.generativeai as genai
        from src.config_manager import SPORT_PRESETS
        from PIL import Image
        
        # Get sport-specific hint from config
        sport_preset = SPORT_PRESETS.get(sport) or SPORT_PRESETS.get('floorball')
        if sport_preset is None:
            raise ValueError(f"Sport preset not found for {sport}")
        hint = getattr(sport_preset, 'hint', f"Analyze this {sport} game footage.")
        
        system_prompt = f"""You are an expert {sport} video analyst with years of experience identifying key game events.

## Your Task
Analyze the provided video frames to find: {instructions}

## Sport-Specific Guidance
{hint}

## What to Look For (Concrete Examples)

✅ REPORT these events:
- **Goal**: Ball/puck completely crosses goal line, net moves, goalkeeper beaten, players raise arms in celebration, teammates hug/high-five
- **Shot**: Player winds up and strikes ball/puck toward goal with force, goalkeeper reacts
- **Save**: Goalkeeper blocks/catches ball/puck, deflects shot away from goal
- **Penalty**: Referee raises arm, points at player, player sent to bench, play stops
- **Assist**: Clear pass directly leading to goal attempt, player sets up scorer

❌ DO NOT report:
- Routine passes or ball possession without shot attempts
- Players simply standing or skating
- Unclear or ambiguous actions

## Output Format (Critical - Follow Exactly)

For EACH event you identify, output valid JSON on a single line:
{{"timestamp": 125.5, "type": "goal", "description": "Player #7 shoots from center, ball enters top right corner of goal, goalkeeper dives but misses, teammates celebrate", "confidence": 0.95}}

**Valid event types**: goal, shot, save, penalty, assist, timeout, turnover

## Boundaries
- ✅ ALWAYS: Provide specific visual evidence in description (player numbers, positions, actions)
- ✅ ALWAYS: Include timestamp based on frame time
- ✅ ALWAYS: Set confidence 0.7-1.0 for clear events, 0.5-0.7 for uncertain
- ⚠️ BE CAUTIOUS: If event is unclear or ambiguous, either lower confidence or skip it
- 🚫 NEVER: Output explanatory text, only JSON objects
- 🚫 NEVER: Report events without clear visual evidence

## Example Good Output
{{"timestamp": 37.5, "type": "goal", "description": "Player in white jersey #10 shoots from 3 meters, ball crosses goal line into bottom left, goalkeeper on ground, players celebrate with arms raised", "confidence": 0.95}}
{{"timestamp": 42.0, "type": "save", "description": "Goalkeeper in red catches high shot with both hands, ball clearly in glove", "confidence": 0.9}}

Analyze ALL frames carefully. Report EVERY event you see with clear evidence. Only output JSON, no other text."""
        
        # Sample frames based on max_frames setting
        sample_frames = frames[::max(1, len(frames) // max_frames)]
        
        # Prepare content with images
        content: List[Any] = [system_prompt]
        
        for i, frame_data in enumerate(sample_frames):
            timestamp = i * frame_interval * (len(frames) / len(sample_frames))
            
            # Load image using PIL
            try:
                import os
                import base64
                from io import BytesIO
                
                if os.path.exists(str(frame_data)):
                    # It's a file path
                    img = Image.open(frame_data)
                else:
                    # It's base64-encoded
                    img_bytes = base64.b64decode(frame_data)
                    img = Image.open(BytesIO(img_bytes))
                
                content.append(f"\n\nFrame at {timestamp:.1f}s:")
                content.append(img)
            except Exception as e:
                print(f"Warning: Could not load frame {frame_data}: {e}")
                continue
        
        try:
            import logging
            logger = logging.getLogger('floorball_llm')
            logger.info(f"Gemini Vision: Analyzing {len(sample_frames)} frames (sampled from {len(frames)} total)")
            
            # Import safety settings enums
            import google.generativeai as genai
            
            # Generate response with proper safety settings
            response = self.client.generate_content(
                content,
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': 4000,  # Increased to allow more events
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
                logger.warning(error_msg)
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
            logger.info(f"Gemini Vision raw response length: {len(response_text)} chars")
            logger.debug(f"Gemini Vision response: {response_text[:500]}...")
            
            events = self.parse_events_from_text(response_text)
            logger.info(f"Gemini Vision: Parsed {len(events)} events from response")
            
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


class PerplexityVisionBackend(VisionBackendMixin):
    """Perplexity Vision backend for video analysis using OpenAI-compatible API."""
    
    def __init__(self, api_key: str, model: str = "sonar"):
        self.api_key = api_key
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai",
                timeout=120.0,
                max_retries=3
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def _analyze_frames_impl(self, frames: List[str], instructions: str, sport: str, frame_interval: float, max_frames: int = 20, time_offset: float = 0.0) -> List[Dict[str, Any]]:
        """Analyze frames using Perplexity Vision API.
        
        Args:
            frames: List of file paths OR base64-encoded strings
        """
        from src.video_tools import encode_image_base64
        from src.config_manager import SPORT_PRESETS
        import os
        
        # Get sport-specific hint from config
        sport_preset = SPORT_PRESETS.get(sport) or SPORT_PRESETS.get('floorball')
        if sport_preset is None:
            raise ValueError(f"Sport preset not found for {sport}")
        hint = getattr(sport_preset, 'hint', f"Analyze this {sport} game footage.")
        
        system_prompt = f"""You are an expert {sport} video analyst with years of experience identifying key game events.

## Your Task
Analyze the provided video frames to find ONLY: {instructions}

IMPORTANT: Only detect and report the specific event types mentioned in the instructions above. Do not report other event types.

## Critical Visual Cues for {sport.title()}
{hint}

## Output Format (Critical - Follow Exactly)

For EACH event you identify, output valid JSON on a single line:
{{"timestamp": 125.5, "type": "goal", "description": "Player #7 shoots from center, ball enters top right corner of goal, goalkeeper dives but misses, teammates celebrate", "confidence": 0.95}}

**Valid event types**: goal, shot, save, penalty, assist, timeout, turnover

## Boundaries & Confidence Levels
- ✅ HIGH CONFIDENCE (0.85-1.0): Multiple clear visual indicators present
- ⚠️ MEDIUM CONFIDENCE (0.7-0.85): Most indicators present but some uncertainty
- ⚠️ LOW CONFIDENCE (0.5-0.7): Limited visual evidence
- 🚫 NEVER: Output explanatory text, only JSON objects
- 🚫 NEVER: Report events without clear visual evidence
- 🚫 NEVER: Report event types not mentioned in the instructions

Analyze ALL frames carefully. Report EVERY event you see that matches the requested types with clear evidence."""
        
        # Sample frames based on max_frames setting
        sample_frames = frames[::max(1, len(frames) // max_frames)]
        
        messages = []
        for i, frame_data in enumerate(sample_frames):
            timestamp = i * frame_interval * (len(frames) / len(sample_frames))
            
            # Check if frame_data is a file path or base64 string
            if os.path.exists(frame_data):
                # It's a file path, encode it
                base64_image = encode_image_base64(frame_data)
            else:
                # It's already base64-encoded
                base64_image = frame_data
            
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
            import logging
            logger = logging.getLogger('floorball_llm')
            logger.info(f"Perplexity Vision: Analyzing {len(sample_frames)} frames (sampled from {len(frames)} total) with model {self.model}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            logger.info(f"Perplexity Vision raw response length: {len(content)} chars")
            logger.debug(f"Perplexity Vision response: {content[:500]}...")
            
            events = self.parse_events_from_text(content)
            logger.info(f"Perplexity Vision: Parsed {len(events)} events from response")
            
            # Apply time offset for chunked processing
            if time_offset > 0:
                for event in events:
                    event['timestamp'] += time_offset
            
            return events
        
        except Exception as e:
            print(f"Perplexity Vision error: {e}")
            return []


def get_vision_backend(backend_name: str, api_key: Optional[str] = None, model: Optional[str] = None):
    """Factory function to get vision backend by name."""
    if backend_name == 'openai' and api_key:
        return OpenAIVisionBackend(api_key, model or "gpt-4o-mini")
    elif backend_name == 'gemini' and api_key:
        return GeminiVisionBackend(api_key, model or "gemini-1.5-flash")
    elif backend_name == 'perplexity' and api_key:
        return PerplexityVisionBackend(api_key, model or "sonar")
    elif backend_name == 'simulated' or not api_key:
        return SimulatedVisionBackend()
    else:
        raise ValueError(f"Vision backend '{backend_name}' not supported. Use 'openai', 'gemini', 'perplexity', or 'simulated'.")
