"""Flask web application for floorball video analysis."""
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
import sys
from pathlib import Path
import tempfile
import os
import json
import yaml
import time
import queue
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_manager import load_config, SPORT_PRESETS
from src.llm_backends_enhanced import SimulatedLLM, HuggingFaceBackend, OllamaBackend
from src.analysis_enhanced import Analyzer
from src.cache import LLMCache
from src.logger import Logger
from src.vision_backends import get_vision_backend
from src.video_tools import prepare_clips

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize global resources
config = load_config()
cache = LLMCache(enabled=config.cache_enabled)
logger = Logger.get_logger()

# Progress tracking
progress_queues = {}
cancelled_tasks = {}  # Track cancelled tasks


def create_progress_queue(task_id):
    """Create a queue for progress updates."""
    progress_queues[task_id] = queue.Queue()
    return progress_queues[task_id]


def send_progress(task_id, step, total_steps, message):
    """Send progress update to queue."""
    if task_id in progress_queues:
        progress_queues[task_id].put({
            'step': step,
            'total': total_steps,
            'message': message,
            'progress': int((step / total_steps) * 100) if total_steps > 0 else 0
        })


def cleanup_progress_queue(task_id):
    """Remove progress queue after completion."""
    if task_id in progress_queues:
        del progress_queues[task_id]


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/settings')
def settings():
    """Settings page."""
    return render_template('settings.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        analyzer = Analyzer(config=config, cache=cache, logger=logger)
        health_status = analyzer.health_check()
        return jsonify({
            'status': 'healthy',
            'backend': health_status['backend'],
            'backend_healthy': health_status['backend_healthy'],
            'cache': health_status['cache_stats']
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/api/analyze/stop/<task_id>', methods=['POST'])
def analyze_stop(task_id):
    """Stop an in-progress analysis and return partial results."""
    cancelled_tasks[task_id] = True
    logger.info(f"Analysis stop requested for task {task_id}")
    return jsonify({'success': True, 'message': 'Stop signal sent'})





@app.route('/api/analyze/start', methods=['POST'])
def analyze_start():
    """Start video analysis with progress tracking."""
    try:
        # Generate task ID
        task_id = str(int(time.time() * 1000))
        
        # Extract data from request
        video_file = request.files.get('video')
        if not video_file:
            return jsonify({'error': 'No video file provided'}), 400
            
        instructions = request.form.get('instructions', 'Find all goals, shots, and penalties')
        backend_name = request.form.get('backend', 'simulated')
        sport = request.form.get('sport', 'floorball')
        
        # Save video file temporarily
        temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{task_id}_{video_file.filename}')
        video_file.save(temp_video_path)
        
        # Cache video path for later clip generation
        video_cache[task_id] = temp_video_path
        
        # Create progress queue BEFORE starting thread
        create_progress_queue(task_id)
        
        def process_video():
            video_path = temp_video_path
            try:
                logger.info(f"=== Starting video analysis for task {task_id} ===")
                logger.info(f"Backend: {backend_name}, Sport: {sport}")
                logger.info(f"Video: {video_path}")
                logger.info(f"Instructions: {instructions}")
                
                # Step 1: Video uploaded
                send_progress(task_id, 1, 5, 'Video uploaded successfully')
                time.sleep(0.1)  # Small delay to ensure message is queued
                
                # Step 2: Initialize backend
                send_progress(task_id, 2, 5, f'Initializing {backend_name} backend...')
                logger.info(f"Initializing {backend_name} backend...")
                api_key = None
                model = None
                if backend_name == 'openai':
                    api_key = os.getenv('OPENAI_API_KEY', '')
                elif backend_name == 'gemini':
                    api_key = os.getenv('GEMINI_API_KEY', '')
                    model = config.gemini_model
                    logger.info(f"Using Gemini model: {model}")
                    
                    # Check if API key is set
                    if not api_key:
                        logger.error("GEMINI_API_KEY not set in environment")
                        raise ValueError("GEMINI_API_KEY not set in .env file. Please add your API key.")
                
                vision_backend = get_vision_backend(backend_name, api_key, model)
                logger.info(f"Backend initialized: {type(vision_backend).__name__}")
                
                # Step 3: Extract frames
                send_progress(task_id, 3, 5, 'Extracting video frames...')
                logger.info("Extracting video frames...")
                from src.video_tools import get_video_duration
                duration = get_video_duration(video_path)
                logger.info(f"Video duration: {duration}s")
                
                # Step 4: Analyze with AI
                send_progress(task_id, 4, 5, f'Analyzing {int(duration)}s video with AI...')
                logger.info(f"Starting AI analysis of {int(duration)}s video...")
                
                # Define progress callback for chunk updates
                def chunk_progress(message):
                    # Send chunk progress as step 4 updates
                    send_progress(task_id, 4, 5, message)
                
                # Define cancellation checker
                def is_cancelled():
                    return cancelled_tasks.get(task_id, False)
                
                result = vision_backend.analyze_video_frames(video_path, instructions, sport, progress_callback=chunk_progress, config=config, is_cancelled_callback=is_cancelled)
                
                events = result.get('events', [])
                meta = result.get('meta', {})
                was_cancelled = meta.get('cancelled', False)
                
                if was_cancelled:
                    logger.info(f"Analysis stopped by user. Processed {meta.get('chunks_completed', 0)}/{meta.get('total_chunks', 0)} chunks. Found {len(events)} events")
                else:
                    logger.info(f"Analysis complete. Found {len(events)} events")
                
                if len(events) == 0 and not was_cancelled:
                    logger.warning("WARNING: No events detected in video!")
                    logger.warning(f"Instructions were: {instructions}")
                    logger.warning("Consider: 1) Adjusting instructions, 2) Checking video quality, 3) Trying different backend")
                
                # Check if there was an error in meta
                if 'error' in meta:
                    logger.error(f"Error in meta: {meta['error']}")
                    raise Exception(f"Video analysis error: {meta['error']}")
                
                # Step 5: Generate clips
                send_progress(task_id, 5, 5, f'Generating {len(events)} video clips...')
                logger.info(f"Preparing to generate {len(events)} clips...")
                clips = []
                if events:
                    clips_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'clips')
                    clips = prepare_clips(events, video_path, clips_dir)
                
                # Send completion
                completion_msg = 'Stopped - Partial Results' if was_cancelled else 'Complete!'
                send_progress(task_id, 5, 5, completion_msg)
                logger.info(f"=== Analysis {'stopped' if was_cancelled else 'complete'} for task {task_id} ===")
                logger.info(f"Total events: {len(events)}")
                logger.info(f"Processing time: {meta.get('processing_ms', 0)}ms")
                logger.info(f"Cost: ${meta.get('cost_usd', 0):.6f}")
                
                # Store result
                progress_queues[task_id].put({
                    'complete': True,
                    'result': {
                        'success': True,
                        'events': events,
                        'clips': [os.path.basename(c) for c in clips],
                        'timestamps': [e.get('timestamp', 0) for e in events],
                        'meta': meta,
                        'cache_hit': False,
                        'cancelled': was_cancelled
                    }
                })
                
                # Clean up cancellation flag
                if task_id in cancelled_tasks:
                    del cancelled_tasks[task_id]
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"=== Video analysis FAILED for task {task_id} ===")
                logger.error(f"Error: {error_message}", exc_info=True)
                
                # Send error to progress stream
                if task_id in progress_queues:
                    progress_queues[task_id].put({
                        'error': error_message,
                        'step': 'failed',
                        'message': f'Error: {error_message}'
                    })
            
            finally:
                # Keep video file for clip generation
                # It will be cleaned up when the server restarts or manually
                pass
        
        # Start processing in background thread
        thread = threading.Thread(target=process_video)
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id, 'status': 'started'})
    
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/progress/<task_id>')
def analyze_progress(task_id):
    """Stream progress updates via Server-Sent Events."""
    def generate():
        if task_id not in progress_queues:
            yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
            return
        
        q = progress_queues[task_id]
        
        try:
            while True:
                try:
                    update = q.get(timeout=30)
                    yield f"data: {json.dumps(update)}\n\n"
                    
                    # Check if complete or error
                    if 'complete' in update or 'error' in update:
                        break
                        
                except queue.Empty:
                    # Send keepalive
                    yield f"data: {json.dumps({'keepalive': True})}\n\n"
        
        finally:
            cleanup_progress_queue(task_id)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/backend/health/<backend_name>', methods=['GET'])
def backend_health(backend_name):
    """Check health of specific backend."""
    try:
        if backend_name == 'simulated':
            backend = SimulatedLLM()
        elif backend_name == 'ollama':
            backend = OllamaBackend()
        elif backend_name == 'huggingface':
            backend = HuggingFaceBackend()
        else:
            return jsonify({'error': 'Unknown backend'}), 400
        
        healthy = backend.health_check()
        return jsonify({'backend': backend_name, 'healthy': healthy})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics."""
    stats = cache.get_stats()
    return jsonify(stats)


@app.route('/api/cache/clear', methods=['POST'])
def cache_clear():
    """Clear cache."""
    try:
        cache.clear()
        return jsonify({'success': True, 'message': 'Cache cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    return jsonify(config.to_dict())


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration and save to config.yaml."""
    try:
        new_config = request.get_json()
        
        # Update config object
        config.llm_backend = new_config.get('llm_backend', config.llm_backend)
        config.gemini_model = new_config.get('gemini_model', config.gemini_model)
        config.sport = new_config.get('sport', config.sport)
        config.cache_enabled = new_config.get('cache_enabled', config.cache_enabled)
        config.cache_dir = new_config.get('cache_dir', config.cache_dir)
        config.clip_output_dir = new_config.get('clip_output_dir', config.clip_output_dir)
        config.max_workers = new_config.get('max_workers', config.max_workers)
        config.ollama_endpoint = new_config.get('ollama_base_url', config.ollama_endpoint)
        
        # Save to config.yaml
        config_path = Path(__file__).parent / 'config.yaml'
        with open(config_path, 'r') as f:
            import yaml
            config_data = yaml.safe_load(f)
        
        # Update values
        config_data['llm_backend'] = config.llm_backend
        config_data['gemini_model'] = config.gemini_model
        config_data['sport'] = config.sport
        config_data['cache_enabled'] = config.cache_enabled
        config_data['cache_dir'] = config.cache_dir
        config_data['clip_output_dir'] = config.clip_output_dir
        config_data['max_workers'] = config.max_workers
        config_data['ollama_endpoint'] = config.ollama_endpoint
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        # Update cache if cache settings changed
        global cache
        cache = LLMCache(enabled=config.cache_enabled)
        
        return jsonify({'success': True, 'message': 'Configuration updated'})
    
    except Exception as e:
        logger.error(f"Failed to update config: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/reset', methods=['POST'])
def reset_config():
    """Reset configuration to defaults."""
    try:
        # Load default config
        from src.config_manager import AppConfig
        default_config = AppConfig()
        
        # Update current config
        config.llm_backend = default_config.llm_backend
        config.gemini_model = default_config.gemini_model
        config.sport = default_config.sport
        config.cache_enabled = default_config.cache_enabled
        config.cache_dir = default_config.cache_dir
        config.clip_output_dir = default_config.clip_output_dir
        config.max_workers = default_config.max_workers
        config.ollama_endpoint = default_config.ollama_endpoint
        
        # Save defaults to config.yaml
        config_path = Path(__file__).parent / 'config.yaml'
        with open(config_path, 'r') as f:
            import yaml
            config_data = yaml.safe_load(f)
        
        config_data['llm_backend'] = config.llm_backend
        config_data['gemini_model'] = config.gemini_model
        config_data['sport'] = config.sport
        config_data['cache_enabled'] = config.cache_enabled
        config_data['cache_dir'] = config.cache_dir
        config_data['clip_output_dir'] = config.clip_output_dir
        config_data['max_workers'] = config.max_workers
        config_data['ollama_endpoint'] = config.ollama_endpoint
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        return jsonify({'success': True, 'message': 'Configuration reset to defaults'})
    
    except Exception as e:
        logger.error(f"Failed to reset config: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/sports', methods=['GET'])
def get_sports():
    """Get available sport presets."""
    sports = {}
    for name, preset in SPORT_PRESETS.items():
        sports[name] = {
            'name': preset.name,
            'event_types': preset.event_types,
            'clip_padding_before': preset.clip_padding_before,
            'clip_padding_after': preset.clip_padding_after
        }
    return jsonify(sports)


# Store video paths temporarily for clip generation
video_cache = {}


@app.route('/api/clips/generate', methods=['POST'])
def generate_clips():
    """Generate video clips from events."""
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        events = data.get('events', [])
        
        if not task_id or not events:
            return jsonify({'error': 'Missing task_id or events'}), 400
        
        # Check if we have the video cached
        if task_id not in video_cache:
            return jsonify({'error': 'Video not found. Please analyze a new video.'}), 404
        
        video_path = video_cache[task_id]
        if not Path(video_path).exists():
            return jsonify({'error': 'Video file no longer exists'}), 404
        
        # Generate clips
        clips_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'clips')
        os.makedirs(clips_dir, exist_ok=True)
        
        clips = prepare_clips(events, video_path, clips_dir)
        
        # Format response with timestamps
        clips_info = []
        for i, clip_path in enumerate(clips):
            event = events[i] if i < len(events) else {}
            timestamp = event.get('timestamp', 0)
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            
            clips_info.append({
                'filename': os.path.basename(clip_path),
                'timestamp': f"{mins}:{str(secs).zfill(2)}",
                'event_type': event.get('type', 'unknown')
            })
        
        return jsonify({
            'success': True,
            'clips': clips_info,
            'total': len(clips)
        })
        
    except Exception as e:
        logger.error(f"Failed to generate clips: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/clips/download/<filename>')
def download_clip(filename):
    """Download a generated clip."""
    try:
        clips_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'clips')
        clip_path = os.path.join(clips_dir, filename)
        
        if not Path(clip_path).exists():
            return jsonify({'error': 'Clip not found'}), 404
        
        return send_file(clip_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Failed to download clip: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/clips/concatenate', methods=['POST'])
def concatenate_clips_endpoint():
    """Concatenate selected clips into a single video."""
    try:
        from src.video_tools import concatenate_clips
        
        data = request.get_json()
        selected_clips = data.get('clips', [])  # List of clip filenames
        
        if not selected_clips:
            return jsonify({'error': 'No clips selected'}), 400
        
        clips_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'clips')
        
        # Build full paths for selected clips
        clip_paths = []
        for clip_filename in selected_clips:
            clip_path = os.path.join(clips_dir, clip_filename)
            if Path(clip_path).exists():
                clip_paths.append(clip_path)
        
        if not clip_paths:
            return jsonify({'error': 'No valid clips found'}), 404
        
        # Generate output filename with timestamp
        import time
        output_filename = f"highlight_reel_{int(time.time())}.mp4"
        output_path = os.path.join(clips_dir, output_filename)
        
        # Concatenate clips
        logger.info(f"Concatenating {len(clip_paths)} clips into {output_filename}")
        success = concatenate_clips(clip_paths, output_path)
        
        if not success:
            return jsonify({'error': 'Failed to concatenate clips'}), 500
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'clips_count': len(clip_paths)
        })
        
    except Exception as e:
        logger.error(f"Failed to concatenate clips: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
