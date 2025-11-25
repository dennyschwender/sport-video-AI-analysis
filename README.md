# Floorball LLM Analysis

**AI-powered video game analysis tool** for extracting events from sports videos and generating clips. Upload game footage, describe what events to find, and get timestamped clips with highlight reels.

## Core Features

🎥 **Video Analysis**
- **Two modes:** Full upload or Local processing (no upload!)
- Upload game videos (up to 5GB, supports MP4/MKV/AVI)
- **Local mode:** Extract frames in browser, upload only frames (~2-5MB instead of 2GB+)
- **Time range filtering:** Analyze only specific portions (e.g., 00:01:30-00:05:00 or 90-300 seconds)
- AI-powered frame analysis with GPT-4o Vision or Gemini
- Custom instructions: "Find all goals and ball losses"
- Real-time progress tracking with 5-step progress bar
- **Execution timer:** Shows elapsed time during analysis (⏱️ Elapsed: 125s)
- **Chunked processing for long videos** - handles full games (2+ hours)
- **Parallel chunk analysis** (4x faster for long videos)
- Automatic event detection with timestamps
- On-demand clip generation with download links
- **Interactive stop button** to cancel analysis and **keep partial results** from processed chunks
- **Field locking:** All inputs disabled during analysis (except stop button)

🤖 **Multiple Vision Backends**
- **OpenAI GPT-4o Vision** - High accuracy video frame analysis
- **Google Gemini Vision** (1.5 Flash/Pro, 2.0 Flash) - Fast & affordable
- **Simulated** - Free offline testing without API costs

📊 **Event Detection**
- Goals, assists, shots, saves, penalties, turnovers, timeouts
- Confidence scores for each detection
- **Advanced confidence filtering:** Use comparison operators (>82%, >=90, <50, <=30)
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

### Testing

```powershell
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_new_features.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

> **Status:** `pytest tests` is run after every change; latest execution (2025-11-25) reported 132 passed and 1 skipped.

**Test Coverage:** 133 tests total (132 passing ✅, 1 skipped)

Test suites include:
- **Core functionality:** Event detection, clipping, backends (89 tests)
- **User features:** Time filtering, confidence operators, UI state (17 tests)
- **Session features:** Select all checkbox, filtered exports, config improvements (23 tests)
  - Select all/none checkbox with filter awareness
  - Filtered exports (timestamps, clips, highlight reels)
  - Max frames auto-calculation from TPM limits
  - Clip padding configuration from sport presets
  - Enhanced AI prompting with detailed visual indicators
  - Video upload endpoint for local mode
  - FFmpeg API compatibility fixes
  - Integration scenarios and workflows

## Usage Guide

### Quick Feature Reference

#### ⏱️ Time Range Filtering (NEW)
Analyze only specific portions of your video:
```
From: 00:01:30  (or just 90 for seconds)
To:   00:05:00  (or just 300 for seconds)
```
**Use cases:**
- Skip intros/outros: Start at 0:30, end 10 seconds before video ends
- Single period: Analyze just 2nd period of game (20:00-40:00)
- Specific plays: Focus on last 5 minutes for comeback analysis
- **Cost savings:** 50-90% reduction for partial analysis

#### 🔍 Advanced Confidence Filtering (NEW)
Use comparison operators to filter events:
```
>82     Show events with confidence greater than 82%
>=90    Show events with confidence 90% or higher
<50     Show events with confidence less than 50%
<=30    Show events with confidence 30% or lower
```
**Use cases:**
- Find reliable detections: `>85`
- Review questionable events: `<60`
- Quality threshold: `>=90`

#### ⏱️ Execution Timer (NEW)
See how long your analysis is taking in real-time:
```
⏱️ Elapsed: 125s
```
- Updates every second
- Helps estimate remaining time
- Useful for cost calculations

#### 🔒 Field Locking (NEW)
All inputs automatically disabled during analysis:
- Prevents accidental configuration changes
- Visual feedback (grayed out, different cursor)
- Stop button remains enabled
- Fields re-enabled when complete

### Two Analysis Modes

#### Mode 1: Standard Upload (Current Page)
- Upload full video file to server
- Best for: Server with good upload speed, Docker deployment
- Clips generated automatically on server

#### Mode 2: Local Processing (Click "💻 Local Mode")
- **Video stays on your computer** - no upload!
- Frames extracted in your browser using Canvas API
- Only frames sent to AI (~2-5MB instead of 2GB+)
- **100x faster "upload"** - 5 seconds vs 5 minutes
- Download timestamps as JSON
- Generate clips locally using ffmpeg
- **Now supports highlight reels!** (Fixed)

**When to use Local Mode:**
- Large video files (>1GB)
- Slow upload speed
- Privacy concerns
- Want to keep original video quality
- Have ffmpeg installed locally

### How to Analyze Videos (Standard Mode)

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
        - Parallel processing (2-4 chunks at once) for 4x speedup
     5. Complete! Events detected
   - **Stop analysis anytime** with ⛔ Stop button - **keeps partial results from processed chunks!**

6. **View Results:**
   - See detected events in table with timestamps
   - If stopped early, see notice: "Analysis Stopped - Showing partial results from 28/44 chunks"
   - **Partial results still usable**: Generate clips and create highlight reels from detected events
   - Click "Generate Clips" to create individual video clips

7. **Create Highlight Reel:**
   - Select clips with checkboxes (or "Select All")
   - Click "🎬 Create Highlight Reel"
   - Download the compiled `highlight_reel_*.mp4` file

### Stop Button with Partial Results

**Problem:** Long video analysis takes 4-5 minutes. If you stop at 3 minutes, you lose all progress.

**Solution:** Stop button now preserves work:
- Click ⛔ Stop button anytime during analysis
- **All completed chunks are saved** (e.g., 28 out of 44 chunks)
- Results show: "Analysis Stopped - Partial results from 28/44 chunks"
- Events detected so far are fully usable
- Generate clips and highlight reels from partial results
- No wasted API calls or processing time

**Example Scenario:**
```
2.4-hour game starts processing (44 chunks total)
✓ Chunks 1-28 complete: 23 events detected
User clicks Stop at chunk 29
⛔ Analysis stopped - 23 events saved
✓ Generate clips from 23 events
✓ Create highlight reel from first ~60 minutes of game
✓ Saved ~2 minutes of processing time
✓ Saved API costs for remaining 16 chunks
```

### Long Video Analysis (Full Games)

For videos over ~7 minutes, the system automatically:
- **Splits into chunks** (25 frames each with 50% overlap)
- **Processes chunks in parallel** (configurable workers per backend)
- **Shows chunk progress** in real-time
- **Deduplicates events** from overlapping regions
- **Handles rate limits** automatically with retry logic

**Example:** 2.4-hour game (8765 seconds)
- Splits into ~44 chunks
- Processes 2-4 chunks concurrently (depending on backend)
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

#### Rate Limit Configuration

The system includes automatic rate limit handling to respect API provider limits. Configure in `config.yaml`:

```yaml
# OpenAI Rate Limits (30,000 TPM for gpt-4o, 500,000 TPM for gpt-4o-mini)
max_workers_openai: 2  # Optional override; leave commented to auto-calc (~26 workers at 500 RPM)

# Gemini Rate Limits (15 RPM free tier, higher for paid)
max_workers_gemini: 4  # Use 4-6 for paid tier, 2 for free tier

# Retry settings for 429 (Rate Limit) errors
rate_limit_retry_delay: 40.0  # Seconds to wait before retry
rate_limit_max_retries: 3     # Max retry attempts per chunk
```

**Check your rate limits:**
- **OpenAI:** https://platform.openai.com/settings/organization/limits
- **Gemini:** https://ai.google.dev/pricing

**Adjust based on your API tier:**
- Higher TPM/RPM → Use more workers for faster processing
- Lower TPM/RPM → Use fewer workers to avoid 429 errors
- If you hit rate limits, the system will automatically retry with exponential backoff

### Local Processing Mode (No Upload)

**Access:** Click "💻 Local Mode (No Upload)" from the main page

**Benefits:**
- ✅ **No video upload** - your video never leaves your computer
- ✅ **100x faster** - upload only frames (~2-5MB) instead of full video (~2GB+)
- ✅ **Privacy-friendly** - video data stays local
- ✅ **Same AI analysis** - uses same backends (OpenAI/Gemini)
- ✅ **Generate clips locally** - using ffmpeg on your machine

**Workflow:**
1. Select video file (stays in browser memory)
2. Browser extracts frames using Canvas API (every 8 seconds)
3. Frames converted to JPEG and uploaded to AI backend
4. AI analyzes frames and returns event timestamps
5. Download timestamps as JSON
6. Generate clips locally using ffmpeg commands

**Example:**
```
Video: game.mp4 (2.5GB, 2 hours)
Extracted: 900 frames @ 8s interval
Uploaded: 3.2MB of JPEG frames
Analysis time: 4 minutes
Downloaded: timestamps.json

# Generate clip locally:
ffmpeg -i game.mp4 -ss 125 -t 10 -c copy goal_2min05s.mp4
```

**Requirements for clip generation:**
- ffmpeg installed: `https://ffmpeg.org/download.html`
- Original video file accessible locally
- Timestamps JSON file from analysis

**Limitations:**
- Clips must be generated manually using ffmpeg
- No automatic highlight reel compilation (must concat clips yourself)
- Requires technical comfort with command line

**Best for:**
- Large video files (>1GB)
- Slow internet connections
- Privacy-sensitive content
- Users comfortable with ffmpeg/CLI tools

1. **Get results:**
   - Timestamps for each event (e.g., `2:15, 5:42, 12:08`)
   - Event descriptions with confidence scores
   - Event table with type, time, description, confidence
   - **Filter events** by any column (Type, Time, Description, Confidence)

2. **Generate clips (optional):**
   - Click "Generate X Clips" button in results
   - **Automatic clip generation** (no ffmpeg commands needed!)
   - Download individual clips for each event
   - Clips saved with timestamps in filenames

**Clip Generation Methods:**
The app automatically uses the best available method:
1. **ffmpeg-python** (fastest, requires `pip install ffmpeg-python`)
2. **ffmpeg subprocess** (fast, requires ffmpeg installed)
3. **moviepy** (slower but pure Python, `pip install moviepy`)

**No manual ffmpeg commands needed!** The app handles everything.

### Column Filtering

Filter events in real-time by typing in the filter row below table headers:

**Filter Options:**
- **Type**: Filter by event type (goal, shot, save, penalty)
- **Time**: Filter by timestamp (e.g., "2:15", "0:48")
- **Description**: Search event descriptions
- **Confidence**: Filter by confidence percentage (e.g., "95%")

**Features:**
- Real-time filtering as you type (no submit needed)
- Case-insensitive search
- Multiple filters combine with AND logic
- Active filters highlighted in green
- Shows "Showing X of Y events" count
- Click 🔄 Clear Filters to reset all

**Examples:**
```
Type: goal, Confidence: 9  → High-confidence goals only
Time: 0:                    → All events in first minute
Description: red team       → Events mentioning red team
```

**Available in both modes:**
- 💻 Local Mode (browser-based analysis)
- 📤 Upload Mode (server-based analysis)

#### Advanced Confidence Filtering with Comparison Operators

Filter events by confidence threshold using comparison operators for precise control:

**Supported Operators:**
- `>` - Greater than (e.g., `>75` finds events above 75% confidence)
- `>=` - Greater than or equal (e.g., `>=80` finds 80% and above)
- `<` - Less than (e.g., `<50` finds events below 50% confidence)
- `<=` - Less than or equal (e.g., `<=30` finds 30% and below)
- No operator - Text matching (e.g., `85` or `85%` finds exactly 85%)

**How It Works:**
1. Type operator and number in Confidence filter box
2. Events filter immediately (no submit needed)
3. Percentage sign (`%`) is optional - `>75` and `>75%` work the same
4. Operators work with or without spaces - `>75` and `> 75` both work

**Example Use Cases:**
```
>80     → High-confidence events only (above 80%)
>=90    → Very high confidence (90% and above)
<60     → Review low-confidence events for false positives
<=40    → Flag potentially incorrect detections
85      → Find exactly 85% confidence events
```

**Workflow Example - Creating High-Quality Highlight Reel:**
```
1. Set Confidence filter: >=85
2. Result: Only high-confidence events shown
3. Select events with checkboxes
4. Click "✨ Combined Highlight Reel"
5. Download professionally-curated highlight video
```

### Video Clip Generation and Downloads

After analysis completes, you have **three download options** for working with detected events:

#### 📄 Download Timestamps (TXT)
- Creates text file with all selected events
- Format: `HH:MM:SS - Event Type - Description`
- Use for: Manual review, sharing timestamps, importing to other tools
- Example output:
  ```
  00:02:15 - goal - Red team scores
  00:05:42 - save - Goalkeeper blocks shot
  00:12:08 - penalty - Blue team penalty
  ```

#### 🎬 Download Individual Clips
- Generates separate video clip for each selected event
- Clips include configurable padding (default: 5-10 seconds before/after event)
- Filenames include timestamp and event type: `clip_000_goal_135.mp4`
- All clips download automatically with 500ms delays
- Use for: Frame-by-frame analysis, social media posts, play review

**How It Works:**
1. Select events using checkboxes (or "Select All")
2. Click "🎬 Individual Clips" button
3. Backend generates clips using ffmpeg
4. Clips automatically download one-by-one
5. Find clips in your Downloads folder

**Technical Details:**
- Uses multi-backend video processing (ffmpeg-python → moviepy → ffmpeg-subprocess)
- Clips saved to `uploads/clips/` directory
- Automatic fallback if one clipping method fails
- No re-encoding (fast copy mode when possible)

#### ✨ Download Combined Highlight Reel
- Concatenates all selected clips into single video
- Preserves chronological order (earliest events first)
- No re-encoding between clips (fast, no quality loss)
- Output filename: `highlight_reel_YYYYMMDD_HHMMSS.mp4`
- Use for: Team review sessions, public sharing, game analysis

**How It Works:**
1. Select key moments using checkboxes
2. Click "✨ Combined Highlight Reel" button
3. Backend generates individual clips
4. Clips concatenated using ffmpeg
5. Single highlight reel downloads automatically
6. Ready to share or review

**Workflow Example - Creating Custom Highlight Reel:**
```
1. Analyze full game (2 hours)
2. Filter: Type="goal", Confidence>85
3. Result: 8 high-confidence goals
4. Select all goals with checkbox
5. Click "✨ Combined Highlight Reel"
6. Download: highlight_reel_20241215_143022.mp4 (contains all 8 goals)
7. Share with team or post to social media
```

**Performance Notes:**
- Individual clips: ~1-2 seconds per clip generation
- Concatenation: ~2-5 seconds for 5-10 clips
- Total time: Usually under 30 seconds for typical highlight reel
- Progress feedback shown during generation

### Advanced Features

#### 🎯 Select All/None Checkbox with Filter Awareness

Smart selection checkbox in the event table header that respects active filters:

**Features:**
- **Filter-aware selection:** Only selects/deselects visible (filtered) events
- **Works with all filters:** Event type, confidence, search text
- **Visual feedback:** Shows current selection state
- **Efficient workflow:** Select hundreds of filtered events with one click

**How It Works:**
```
1. Apply filters (e.g., Type="goal", Confidence>85)
2. Click checkbox in table header
3. Only visible filtered events get selected
4. Hidden events remain unaffected
5. Export or download selected events
```

**Example Workflow - Filtering + Bulk Selection:**
```
Scenario: 200 total events, want only high-confidence goals
1. Filter: Type="goal" → Shows 45 goals
2. Filter: Confidence>85 → Shows 12 high-confidence goals
3. Click select all checkbox → 12 goals selected
4. Click "✨ Combined Highlight Reel"
5. Downloads: highlight reel with 12 best goals only
```

#### 📤 Filtered Exports

All export functions now respect active filters and only export selected + visible events:

**Export Types:**
1. **📄 Download Timestamps (TXT)** - Only selected + visible events
2. **🎬 Individual Clips** - Only selected + visible events  
3. **✨ Highlight Reel** - Only selected + visible events

**Workflow Example:**
```
200 total events detected
↓ Filter: Type="goal" → 45 events visible
↓ Filter: Confidence>=80 → 18 events visible
↓ Select 12 events with checkboxes
↓ Click "📄 Download Timestamps"
→ TXT file contains only those 12 selected events (not all 200)
```

**Before vs After:**
- **Before:** Export includes all events, ignoring filters
- **After:** Export only includes selected events that pass filters
- **Benefit:** Precise control over exported content

#### ⚙️ Max Frames Auto-Calculation

System automatically calculates optimal `max_frames` based on API rate limits (TPM/RPM):

**Formula:**
```
max_frames = (TPM / 12 - 2500) / 850
```

**Configured Defaults:**
- **OpenAI gpt-4o-mini** (500,000 TPM): 55 frames
- **OpenAI gpt-4o** (30,000 TPM): 0 frames (chunking disabled)
- **Gemini Flash** (4,000,000 TPM): 388 frames
- **Gemini Pro** (360,000 TPM): 38 frames

**How It Works:**
```yaml
# config.yaml
backends:
  openai:
    model: "gpt-4o-mini"
    tokens_per_minute: 500000  # ← System reads this
    max_frames: 55             # ← Auto-calculated from formula
```

**Benefits:**
- No manual calculation needed
- Prevents rate limit errors
- Optimized for each model's capabilities
- Safe defaults that maximize throughput

**Override:** Set custom `max_frames` in config.yaml to override auto-calculation

#### 🎬 Configurable Clip Padding

Clip duration now configurable per sport via `config.yaml`:

**Configuration:**
```yaml
sport_presets:
  floorball:
    clip_padding_before: 10   # Seconds before event
    clip_padding_after: 5     # Seconds after event
  
  hockey:
    clip_padding_before: 8
    clip_padding_after: 7
```

**Total Clip Duration:**
```
Clip duration = padding_before + padding_after
Floorball: 10s + 5s = 15 second clips
Hockey: 8s + 7s = 15 second clips
```

**Benefits:**
- **Sport-specific timing:** More buildup for floorball (10s before), quicker cuts for hockey
- **Flexible customization:** Adjust per your analysis needs
- **Consistent across app:** All clip generation uses same padding
- **Easy to modify:** Edit config.yaml, no code changes needed

**Example Use Cases:**
- More context: Increase `padding_before` to 15s for play development
- Quick highlights: Decrease both to 3s for rapid-fire compilation
- Slow-motion analysis: Increase `padding_after` to capture full aftermath

#### 🎯 Enhanced AI Prompting

AI vision prompts now include detailed visual indicators for each event type:

**6 Visual Indicators Per Event:**
1. **Primary indicator** - Main visual cue (e.g., "ball crosses goal line")
2. **Player reaction** - Celebration, frustration, arms raised
3. **Goalkeeper action** - Diving, retrieving ball, looking defeated
4. **Crowd/bench reaction** - Standing, cheering, hands on heads
5. **Scoreboard changes** - Score updates (when visible)
6. **Game flow changes** - Faceoff at center, timeout called

**Confidence Level Guidance:**
```
HIGH (0.85-1.0):  All 4+ indicators clearly visible
MEDIUM (0.7-0.85): 2-3 indicators visible
LOW (0.5-0.7):     Only 1-2 indicators visible
```

**Example - Goal Detection:**
```
Look for these visual indicators:
✓ Ball crossing goal line
✓ Players raising arms in celebration
✓ Goalkeeper retrieving ball from net
✓ Opposing team looking deflated
✓ Scoreboard updating (if visible)
✓ Faceoff at center circle after

Confidence: 0.95 (all 6 indicators present)
```

#### 🧠 Goal Confirmation & Annotation (NEW)

The backend now double-checks uncertain goals by re-analyzing the frames around each candidate, and it can log confirmed goals for manual review.

**Workflow:**
1. Goals without clear supporting events (<0.9 confidence) trigger a dense-sampling pass around ±2.5 seconds at 0.25s intervals.
2. Confirmed goals receive at least one supporting indicator (ball crossing the line) and their confidence is bumped to ≥0.8.
3. Goals that only mention scoreboard updates (no visible action) are capped at 0.65 confidence to avoid false positives.
4. Enable annotation mode to append high-confidence goals (`confidence ≥ 0.7`) to `annotations/goals/goal_candidates.jsonl` for future training.

**Configuration (in `config.yaml`):**
```yaml
goal_refinement_enabled: true
goal_refinement_attempts: 2
goal_refinement_window: 2.5
goal_refinement_interval: 0.25
goal_annotation_enabled: false
goal_annotation_dir: annotations/goals
goal_annotation_threshold: 0.7
```

Use the annotation log to build a curated dataset of confirmed goals or to feed into downstream machine‑learning tooling.

**Benefits:**
- **Higher accuracy:** AI checks multiple visual cues
- **Better confidence scores:** Based on number of indicators
- **Fewer false positives:** Won't detect goal from scoreboard alone
- **Clearer reasoning:** AI explains what it saw

**Improved Event Types:**
- Goals: 6 detailed indicators
- Shots: 6 detailed indicators  
- Saves: 6 detailed indicators
- Assists, penalties, turnovers: Enhanced descriptions

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

### GET /api/clips/download/\<filename\>

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
