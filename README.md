# Floorball LLM Analysis

**AI-powered video game analysis tool** for extracting events from sports videos and generating clips. Upload game footage, describe what events to find, and get timestamped clips with highlight reels.

## Core Features

🎥 **Video Analysis**
- Upload game videos (up to 5GB, supports MP4/MKV/AVI)
- AI-powered frame analysis with GPT-4o Vision or Gemini
- Custom instructions: "Find all goals and ball losses"
- Real-time progress tracking with 5-step progress bar
- **Chunked processing for long videos** - handles full games (2+ hours)
- **Parallel chunk analysis** (4x faster for long videos)
- Automatic event detection with timestamps
- On-demand clip generation with download links
- **Interactive stop button** to cancel analysis

🤖 **Multiple Vision Backends**
- **OpenAI GPT-4o Vision** - High accuracy video frame analysis
- **Google Gemini Vision** (1.5 Flash/Pro, 2.0 Flash) - Fast & affordable
- **Simulated** - Free offline testing without API costs

📊 **Event Detection**
- Goals, assists, shots, saves, penalties, turnovers, timeouts
- Confidence scores for each detection
- Team and player identification
- Returns list of timestamps for clipping
- Sport-specific frame sampling (floorball: 8s, hockey: 10s, soccer: 15s)

🎬 **Smart Clipping & Highlight Reels**
- Automatic clip generation using ffmpeg
- Event-specific padding (5-10 seconds before/after)
- **Multi-clip selection with checkboxes**
- **Create custom highlight reels** from selected clips
- Fast concatenation with ffmpeg (no re-encoding)
- Downloadable highlight_reel_*.mp4 files

⚙️ **Production-Ready**
- Flask + Gunicorn for scalable deployment
- Dedicated settings page for configuration management
- Real-time progress updates via Server-Sent Events (SSE)
- **Chunk progress display** for long videos
- Docker & Docker Compose support
- 5GB upload limit for full games
- Comprehensive logging and error handling
- Cost tracking for API usage

🚀 **Performance Features**
- **Parallel chunk processing** - 4 concurrent API calls for 4x speedup
- **Automatic chunking** for videos >400 seconds
- **Smart deduplication** - removes duplicate events from overlapping chunks
- **Optimized frame sampling** - sport-specific intervals

## Prerequisites

- **Python 3.11+**
- **ffmpeg** (required for video processing)

### Install ffmpeg

```powershell
# Windows (using winget)
winget install FFmpeg

# Or download from: https://ffmpeg.org/download.html
```

### LLM Backend Setup

Choose one or more backends based on your needs:

#### 1. Simulated Backend (No Setup Required) - Free
Perfect for testing and development without API costs.

**Setup:** None required, works out of the box!

```powershell
# Just run the app, select "simulated" backend
python app.py
```

#### 2. OpenAI GPT-4o Vision (Recommended for Production)
Analyzes actual video frames with high accuracy.

**Setup:**
1. Get API key from https://platform.openai.com/api-keys
2. Set environment variable:
   ```powershell
   $env:OPENAI_API_KEY="sk-your-key-here"
   ```
3. Or add to `.env` file:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

**Install Python package (optional, for better integration):**
```powershell
pip install openai
```

**Cost:** ~$0.05-0.50 per video (1 frame per 10 seconds)

#### 3. Anthropic Claude 3.5 Sonnet (Text Analysis)
For transcript analysis, not video frames yet.

**Setup:**
1. Get API key from https://console.anthropic.com/
2. Set environment variable:
   ```powershell
   $env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
   ```
3. Install Python package:
   ```powershell
   pip install anthropic
   ```

**Note:** Vision support coming soon.

#### 4. Ollama (Self-Hosted, Free)
Run LLMs locally on your machine. Good for privacy and cost control.

**Setup:**
1. Install Ollama from https://ollama.ai/download
2. Pull a model:
   ```powershell
   ollama pull llama2
   # Or for better results:
   ollama pull llama3.1
   ```
3. Start Ollama server:
   ```powershell
   ollama serve
   ```
4. Set base URL (default is http://localhost:11434):
   ```powershell
   $env:OLLAMA_BASE_URL="http://localhost:11434"
   ```

**Note:** Currently supports text analysis. Vision models coming soon.

#### 5. Google Gemini (Vision Support - Recommended for Long Videos)
Analyze video frames with Gemini 1.5/2.0 Flash or Pro.

**Setup:**
1. Get API key from https://aistudio.google.com/app/apikey
2. Add to `.env` file:
   ```
   GEMINI_API_KEY=your-gemini-key-here
   ```
3. Install Python packages:
   ```powershell
   pip install google-generativeai>=0.3.0 Pillow
   ```

**Cost:** Varies by model (updated Dec 2024)
- **Gemini 1.5 Flash**: $0.075/$0.30 per 1M tokens (input/output) - **Best value**
- **Gemini 1.5 Pro**: $1.25/$5.00 per 1M tokens - Higher quality
- **Gemini 2.0 Flash**: $0.075/$0.30 per 1M tokens - Latest model
- Typical cost: ~$0.02-0.10 per video (much cheaper than GPT-4o)

**Pros:**
- ✅ Large context window (1M+ tokens)
- ✅ Fast inference (especially Flash model)
- ✅ Good vision capabilities
- ✅ Much cheaper than GPT-4o Vision (5-10x less)
- ✅ Works great with parallel chunk processing

**Cons:**
- ⚠️ Safety filters can be overly strict (use Flash over Pro)
- ⚠️ JSON output less consistent than OpenAI

**Recommended Model:**
- For most use cases: `gemini-1.5-flash` or `gemini-2.0-flash`
- For highest quality: `gemini-1.5-pro`
- Don't use `-latest` suffix in config (just `gemini-1.5-flash`)

#### 6. HuggingFace Inference API (Budget Option)
Access various models through HuggingFace API.

**Setup:**
1. Get API token from https://huggingface.co/settings/tokens
2. Set environment variable:
   ```powershell
   $env:HUGGINGFACE_API_KEY="hf_your-token-here"
   ```
3. Install Python package:
   ```powershell
   pip install huggingface_hub
   ```

**Cost:** Very low (~$0.001 per analysis)

### Backend Recommendation

| Use Case | Recommended Backend | Why |
|----------|-------------------|-----|
| **Testing/Development** | Simulated | Free, instant, no setup |
| **Production (Best Quality)** | OpenAI GPT-4o Vision | Highest accuracy for frames |
| **Production (Best Value)** | Gemini 1.5/2.0 Flash | Great quality, 5-10x cheaper |
| **Long Videos (2+ hours)** | Gemini Flash + Parallel Processing | Fast, affordable, handles chunks well |
| **Large Context** | Gemini 1.5 Pro | 1M+ token window |
| **Budget Video Analysis** | Gemini 1.5 Flash | ~$0.02-0.10 per video |
| **Privacy/Self-Hosted** | Ollama | Runs locally, no data sent externally |

**For Full Game Analysis (2+ hours):**
- Use Gemini 1.5 Flash with parallel chunk processing enabled
- Expected processing time: 4-5 minutes (vs 15-20 minutes sequential)
- Cost: ~$0.50-1.50 for full game (vs $5-10 with GPT-4o)

## Quick Start

### Option 1: Docker Production Deployment (Recommended)

```powershell
# Navigate to project
cd floorball_llm

# Copy environment variables template
Copy-Item .env.example -Destination .env
# Edit .env and add your OPENAI_API_KEY
# Edit .env with your API keys (nano .env or notepad .env)

# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f app

# Stop services
docker-compose down
```

Navigate to `http://localhost:5000`.

#### Docker Commands

```powershell
# Build image only
docker build -t floorball-llm .

# Run container manually
docker run -p 5000:5000 `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  -e LLM_BACKEND=simulated `
  -v ${PWD}/cache:/app/cache `
  floorball-llm

# Shell into container
docker exec -it floorball-llm-app /bin/bash
```

### Option 2: Local Development

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Optional: Install LLM backend packages
pip install openai        # For OpenAI GPT-4o Vision
pip install anthropic     # For Claude 3.5
pip install huggingface_hub  # For HuggingFace models
```

#### Configure Environment Variables

The project uses **two configuration files** with clear separation:

1. **`.env`** - Secrets/API keys (never commit to git)
2. **`config.yaml`** - Application settings (safe to commit)

**Setup .env file (API Keys - REQUIRED):**
```powershell
# Copy the template
Copy-Item .env.example -Destination .env

# Edit .env with your favorite editor
notepad .env
# Or: code .env
```

**Edit `.env` file** and add your API keys:
```bash
# API Keys (secrets - keep private!)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
HUGGINGFACE_API_KEY=hf_your-huggingface-token-here
```

**Setup config.yaml (Application Settings - OPTIONAL):**
```powershell
# Copy the template
Copy-Item config.yaml.example -Destination config.yaml

# Edit config.yaml for app preferences
notepad config.yaml
# Or: code config.yaml
```

**Edit `config.yaml`** to customize your setup:
```yaml
# Which backend to use
llm_backend: simulated  # or: openai, anthropic, ollama

# Which sport to analyze
sport: floorball  # or: hockey, soccer

# Model preferences
openai_model: gpt-4o-mini
anthropic_model: claude-3-5-sonnet-20241022

# Processing settings
cache_enabled: true
clip_output_dir: clips
```

**Note:** 
- `.env` is automatically loaded by `python-dotenv` and contains **only secrets**
- `config.yaml` contains **app settings** (models, sport, cache settings)
- API keys in `.env` ALWAYS override config.yaml (for security)
- `.env` is in `.gitignore` - never commit it!
- `config.yaml` is safe to commit (no secrets)

#### Run Flask Production Server

```powershell
# Using gunicorn (recommended for production)
gunicorn --config gunicorn.conf.py app:app

# Using Flask dev server (development only)
$env:FLASK_APP="app.py"
$env:FLASK_ENV="development"
flask run
```

Navigate to `http://localhost:5000`.

#### Run Streamlit Prototype UI

```powershell
python -m streamlit run scripts/web_ui.py
```

Navigate to `http://localhost:8501`.

### Basic CLI Usage

```powershell
# Create a config file (optional)
python -c "from src.config_manager import AppConfig; AppConfig().to_yaml('config.yaml')"

# Run analysis on a video (requires transcript .txt file alongside)
python scripts/run_analysis.py --video path\to\video.mp4
```

### Benchmarking

```powershell
python scripts/benchmark_enhanced.py
```

## Testing

```powershell
pytest tests/test_enhanced.py -v
```

All 38 tests passing ✅

## Usage Guide

### How to Analyze Videos

1. **Start the application:**
   ```powershell
   # Local development
   python app.py
   
   # Or with Docker
   docker-compose up -d
   ```

2. **Upload video at http://localhost:5000**

3. **Describe what to find** in the instructions box:
   - `Find all goals and assists`
   - `Find all ball losses, turnovers, and saves`
   - `Find all penalties and penalty shots`

4. **Configure settings (optional):**
   - Click ⚙️ Settings to choose backend, sport, and models
   - Or use defaults: `gemini` backend with `floorball` sport

5. **Click "🔍 Analyze Video"**
   - Watch real-time progress with 5 steps:
     1. Video uploaded successfully
     2. Backend initialized
     3. Extracting video frames
     4. Analyzing video with AI
        - For long videos (>400s), see chunk progress: "Processing chunk 5/44: 200s-400s → 3 events"
        - Parallel processing (4 chunks at once) for 4x speedup
     5. Complete! Events detected
   - **Stop analysis anytime** with ⛔ Stop button

6. **View Results:**
   - See detected events in table with timestamps
   - Click "Generate Clips" to create individual video clips
   
7. **Create Highlight Reel:**
   - Select clips with checkboxes (or "Select All")
   - Click "🎬 Create Highlight Reel"
   - Download the compiled `highlight_reel_*.mp4` file

### Long Video Analysis (Full Games)

For videos over ~7 minutes, the system automatically:
- **Splits into chunks** (25 frames each with 50% overlap)
- **Processes chunks in parallel** (4 at a time)
- **Shows chunk progress** in real-time
- **Deduplicates events** from overlapping regions

**Example:** 2.4-hour game (8765 seconds)
- Splits into ~44 chunks
- Processes 4 chunks concurrently
- Completes in ~4-5 minutes (vs 15-20 minutes sequential)
- Detects events throughout entire game without missing action

**Progress messages you'll see:**
```
Long video detected (8765.1s). Processing 44 chunks in parallel...
✓ Chunk 1/44: 0s-200s → 3 events
✓ Chunk 2/44: 100s-300s → 2 events
...
Parallel processing complete: 127 raw events → 89 unique events
```
     3. Frames extracted
     4. AI analysis
     5. Complete

6. **Get results:**
   - Timestamps for each event (e.g., `2:15, 5:42, 12:08`)
   - Event descriptions with confidence scores
   - Event table with type, time, description, confidence

7. **Generate clips (optional):**
   - Click "Generate X Clips" button in results
   - Download individual clips for each event
   - Clips saved with timestamps in filenames

### Example Workflows

**Create Goal Highlights:**
```
Instructions: Find all goals
Backend: openai
Result: Timestamped clips of every goal
```

**Analyze Player Performance:**
```
Instructions: Find all shots, goals, assists, and turnovers
Result: Complete player action timeline
```

**Scout Opponent:**
```
Instructions: Find all defensive weaknesses, turnovers, and penalty situations
Result: Pattern analysis for team review
```

### Backend Comparison

| Feature | OpenAI GPT-4o | Gemini 3 Pro | Gemini 1.5 Flash | Simulated | Anthropic | HuggingFace | Ollama |
|---------|---------------|--------------|------------------|-----------|-----------|-------------|--------|
| Analyzes video frames | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ❌ No* | ❌ No | ❌ No* |
| Accuracy | Very High | High | High | N/A | High (text) | Medium | Medium |
| Cost per video | $0.05-0.50 | $0.25-0.75 | $0.02-0.10 | Free | $0.01-0.05 | $0.001 | Free |
| API key required | Yes | Yes | Yes | No | Yes | Yes | No |
| Use case | High accuracy | Latest model | Budget video | Testing | Transcript | Ultra-budget | Self-hosted |

*Vision support coming soon

### Tips for Better Results

✅ **Be specific:** "Find all goals, shots on goal, and penalty shots"  
✅ **Use sport terms:** "Find all goals" not "Find when ball goes in net"  
✅ **Video quality matters:** Higher resolution = better detection  
✅ **Optimize length:** 10-20 min chunks process faster than full games

## Troubleshooting

### "No frames extracted" Error
**Solution:** Install ffmpeg: `winget install FFmpeg`

### "OpenAI API error"
**Cause:** Invalid or missing API key  
**Solution:** 
1. Get API key from https://platform.openai.com/api-keys
2. Set environment variable: `$env:OPENAI_API_KEY="sk-..."`
3. Or add to `.env` file

### "Anthropic API error"
**Solution:** Set API key: `$env:ANTHROPIC_API_KEY="sk-ant-..."`

### "Ollama connection refused"
**Cause:** Ollama server not running  
**Solution:** 
1. Start Ollama: `ollama serve`
2. Verify it's running: `ollama list`
3. Check URL: `$env:OLLAMA_BASE_URL="http://localhost:11434"`

### "Module 'openai' not found"
**Solution:** Install the package: `pip install openai`

### No Events Detected
**Possible causes:**
- Instructions too vague → Be more specific
- Wrong backend → Use `openai` for video, others for transcripts
- Low video quality → Use higher resolution video
- Backend not configured → Check API key is set

### Clips Not Generated
**Solution:** Verify ffmpeg is installed: `ffmpeg -version`

### "Rate limit exceeded" (OpenAI/Anthropic)
**Cause:** Too many API requests  
**Solution:** 
1. Wait a few minutes
2. Upgrade your API plan
3. Use `simulated` backend for testing

## Cost Estimation

| Video Length | Frames Analyzed | Estimated Cost |
|--------------|----------------|----------------|
| 5 minutes    | ~30 frames     | $0.05 - $0.10  |
| 15 minutes   | ~90 frames     | $0.15 - $0.25  |
| 30 minutes   | ~180 frames    | $0.30 - $0.50  |
| 60 minutes   | ~360 frames    | $0.60 - $1.00  |

*System samples 1 frame per 10 seconds. Max 20 frames per API call.*

## API Reference

### POST /api/analyze/start

Upload video and start analysis with progress tracking.

**Returns:** `{"task_id": "1234567890", "status": "started"}`

### GET /api/analyze/progress/<task_id>

Stream progress updates via Server-Sent Events (SSE).

**Returns:** Real-time progress updates with step, message, and percentage.

### POST /api/clips/generate

Generate video clips from detected events.

**Request:**
```json
{
  "task_id": "1234567890",
  "events": [{"type": "goal", "timestamp": 125.5, ...}]
}
```

**Response:**
```json
{
  "success": true,
  "clips": [
    {"filename": "clip_000.mp4", "timestamp": "2:05", "event_type": "goal"}
  ],
  "total": 1
}
```

### GET /api/clips/download/<filename>

Download a generated clip file.

### POST /api/analyze (Legacy)

Upload video and get timestamped events (old endpoint, kept for compatibility).

**Request:**
```powershell
$form = @{
    video = Get-Item "game.mp4"
    instructions = "Find all goals and assists"
    backend = "openai"
    sport = "floorball"
}
Invoke-WebRequest -Uri http://localhost:5000/api/analyze -Method POST -Form $form
```

**Response:**
```json
{
  "success": true,
  "events": [
    {
      "type": "goal",
      "timestamp": 125.5,
      "description": "Goal scored",
      "confidence": 0.92
    }
  ],
  "timestamps": [125.5, 340.2],
  "clips": ["clip_000_goal_125.mp4"],
  "meta": {
    "processing_ms": 45000,
    "frames_analyzed": 20,
    "cost_usd": 0.15
  }
}
```

## Configuration

### Environment Variables

Create `.env` file from template:
```powershell
Copy-Item .env.example -Destination .env
```

**Required for OpenAI:**
```
OPENAI_API_KEY=sk-your-key-here
```

**Optional:**
```
LLM_BACKEND=simulated
SPORT=floorball
GUNICORN_WORKERS=4
LOG_LEVEL=info
```

### Sport Presets

- **Floorball:** goal, assist, shot, save, penalty, timeout
- **Hockey:** + icing, offside
- **Soccer:** + corner, freekick, yellow_card, red_card

## Advanced Usage

### Python API

```python
from src.vision_backends import get_vision_backend
import os

backend = get_vision_backend('openai', os.getenv('OPENAI_API_KEY'))
result = backend.analyze_video_frames(
    video_path='match.mp4',
    instructions='Find all goals and penalties',
    sport='floorball'
)

for event in result['events']:
    print(f"{event['type']} at {event['timestamp']}s")
```

### Batch Processing

```powershell
# Analyze multiple videos
Get-ChildItem *.mp4 | ForEach-Object {
    $form = @{
        video = $_
        instructions = "Find all goals"
        backend = "openai"
    }
    Invoke-WebRequest -Uri http://localhost:5000/api/analyze -Method POST -Form $form
}
```

## Project Structure

```
floorball_llm/
├── src/
│   ├── vision_backends.py      # Video frame analysis
│   ├── video_tools.py          # ffmpeg integration
│   ├── llm_backends_enhanced.py # LLM backends
│   ├── analysis_enhanced.py    # Main analyzer
│   ├── cache.py                # Response caching
│   └── config_manager.py       # Configuration
├── templates/
│   └── index.html              # Web UI
├── app.py                      # Flask application
├── Dockerfile                  # Container image
└── docker-compose.yml          # Stack deployment
```

## Contributing

Contributions welcome! Recent additions:
- ✅ Real-time progress tracking with SSE
- ✅ Dedicated settings page for configuration
- ✅ On-demand clip generation with download links
- ✅ Gemini Vision backend integration
- ✅ Cost tracking for multiple backends

Areas for future improvement:
- Add Claude Vision backend
- Add audio transcription (Whisper)
- Mobile-responsive UI improvements
- Batch processing CLI
- Clip preview thumbnails

## License

MIT License - See LICENSE file for details
