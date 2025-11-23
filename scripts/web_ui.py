"""Streamlit web interface for floorball video analysis."""
import streamlit as st
import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_manager import load_config, SPORT_PRESETS
from src.llm_backends_enhanced import SimulatedLLM, HuggingFaceBackend, OllamaBackend
from src.analysis_enhanced import Analyzer
from src.cache import LLMCache
from src.logger import Logger


st.set_page_config(
    page_title="Floorball LLM Analysis",
    page_icon="🏒",
    layout="wide",
    # Allow large video files up to 5GB
    menu_items={
        'About': "Floorball LLM Analysis - AI-powered game analysis"
    }
)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = load_config()
if 'cache' not in st.session_state:
    st.session_state.cache = LLMCache(enabled=st.session_state.config.cache_enabled)
if 'logger' not in st.session_state:
    st.session_state.logger = Logger.get_logger()


def main():
    st.title("🏒 Floorball Game Analysis")
    st.markdown("Analyze floorball game videos using LLM to extract events and create clips")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Backend selection
        backend = st.selectbox(
            "LLM Backend",
            ["simulated", "openai", "anthropic", "huggingface", "ollama"],
            index=0
        )
        
        # Sport selection
        sport = st.selectbox(
            "Sport",
            list(SPORT_PRESETS.keys()),
            index=0
        )
        
        # Cache settings
        cache_enabled = st.checkbox("Enable Caching", value=st.session_state.config.cache_enabled)
        
        if cache_enabled:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Cache"):
                    st.session_state.cache.clear()
                    st.success("Cache cleared!")
            with col2:
                stats = st.session_state.cache.get_stats()
                if stats.get("enabled"):
                    st.metric("Cached Items", stats.get("size", 0))
        
        st.divider()
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            dedup_window = st.slider("Deduplication Window (s)", 0.0, 10.0, 5.0, 0.5)
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)
            clip_padding_before = st.number_input("Clip Padding Before (s)", 0, 30, 5)
            clip_padding_after = st.number_input("Clip Padding After (s)", 0, 30, 8)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📹 Analyze Video", "📊 Event Timeline", "🎬 Create Compilations"])
    
    with tab1:
        st.header("Video Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video/transcript upload
            upload_method = st.radio("Input Method", ["Upload Video", "Paste Transcript"])
            
            if upload_method == "Upload Video":
                video_file = st.file_uploader(
                    "Upload video file (supports up to 5GB)", 
                    type=["mp4", "avi", "mov", "mkv", "flv"]
                )
                if video_file:
                    # Show file info
                    file_size_mb = video_file.size / (1024 * 1024)
                    st.info(f"📁 File: {video_file.name} ({file_size_mb:.1f} MB)")
                    
                    if file_size_mb < 500:  # Only preview smaller files
                        st.video(video_file)
                    else:
                        st.warning(f"⚠️ Large file ({file_size_mb:.1f} MB) - preview disabled for performance")
                    
                    st.warning("Note: Transcript extraction not yet implemented. Please use 'Paste Transcript' for now.")
            else:
                transcript = st.text_area(
                    "Paste game commentary transcript",
                    height=300,
                    placeholder="0:00 Kickoff by team A\n0:45 Shot on goal by player #7...",
                    help="Enter the game commentary with timestamps"
                )
        
        with col2:
            st.subheader("Backend Info")
            
            # Create backend instance
            try:
                if backend == "simulated":
                    backend_instance = SimulatedLLM()
                    st.success("✅ Simulated backend ready")
                elif backend == "ollama":
                    backend_instance = OllamaBackend()
                    health = backend_instance.health_check()
                    if health:
                        st.success("✅ Ollama connected")
                    else:
                        st.error("❌ Ollama not available")
                elif backend == "huggingface":
                    api_key = st.text_input("HuggingFace API Key", type="password")
                    backend_instance = HuggingFaceBackend(api_key=api_key if api_key else None)
                    if api_key:
                        st.success("✅ HuggingFace configured")
                    else:
                        st.warning("⚠️ No API key provided")
                else:
                    st.warning(f"⚠️ {backend} backend requires configuration")
                    backend_instance = SimulatedLLM()
            
            except Exception as e:
                st.error(f"Error: {e}")
                backend_instance = SimulatedLLM()
        
        # Analyze button
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if upload_method == "Paste Transcript" and transcript:
                with st.spinner("Analyzing commentary..."):
                    try:
                        # Check cache first
                        model_name = getattr(backend_instance, 'model', 'unknown')
                        cached = st.session_state.cache.get(backend, model_name, transcript)
                        
                        if cached:
                            result = cached
                            st.info("ℹ️ Result loaded from cache")
                        else:
                            result = backend_instance.analyze_commentary(transcript)
                            st.session_state.cache.set(backend, model_name, transcript, result)
                        
                        st.session_state['analysis_result'] = result
                        st.session_state['transcript'] = transcript
                        
                        # Display results
                        events = result.get("events", [])
                        meta = result.get("meta", {})
                        
                        st.success(f"✅ Found {len(events)} events")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Events Found", len(events))
                        with col2:
                            st.metric("Processing Time", f"{meta.get('processing_ms', 0)}ms")
                        with col3:
                            st.metric("Cost", f"${meta.get('cost_usd', 0):.6f}")
                        with col4:
                            if 'input_tokens' in meta:
                                st.metric("Tokens", f"{meta['input_tokens']}→{meta.get('output_tokens', 0)}")
                        
                        # Events table
                        if events:
                            st.subheader("Detected Events")
                            st.dataframe(
                                events,
                                use_container_width=True,
                                height=400
                            )
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            else:
                st.warning("Please provide a transcript to analyze")
    
    with tab2:
        st.header("Event Timeline")
        
        if 'analysis_result' in st.session_state:
            events = st.session_state['analysis_result'].get("events", [])
            
            if events:
                # Filter events
                event_types = list(set(e.get("type", "unknown") for e in events))
                selected_types = st.multiselect("Filter by Event Type", event_types, default=event_types)
                
                filtered_events = [e for e in events if e.get("type") in selected_types]
                
                # Timeline visualization (simple)
                st.subheader(f"Timeline ({len(filtered_events)} events)")
                
                for event in sorted(filtered_events, key=lambda e: e.get("timestamp", 0)):
                    timestamp = event.get("timestamp", 0)
                    event_type = event.get("type", "unknown")
                    description = event.get("description", "")
                    confidence = event.get("confidence")
                    
                    col1, col2, col3 = st.columns([1, 2, 6])
                    with col1:
                        st.text(f"{int(timestamp//60)}:{int(timestamp%60):02d}")
                    with col2:
                        st.badge(event_type.upper())
                    with col3:
                        conf_text = f" (conf: {confidence:.2f})" if confidence else ""
                        st.text(description + conf_text)
            else:
                st.info("No events to display. Run an analysis first.")
        else:
            st.info("No analysis results yet. Analyze a video in the first tab.")
    
    with tab3:
        st.header("Create Compilations")
        
        if 'analysis_result' in st.session_state:
            events = st.session_state['analysis_result'].get("events", [])
            
            if events:
                compilation_type = st.radio(
                    "Compilation Type",
                    ["Highlight Reel", "By Event Type", "By Player", "By Team"]
                )
                
                if compilation_type == "Highlight Reel":
                    st.subheader("Create Highlight Reel")
                    min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.1)
                    include_types = st.multiselect(
                        "Include Event Types",
                        list(set(e.get("type") for e in events)),
                        default=["goal"]
                    )
                    
                    if st.button("Generate Highlight Reel"):
                        filtered = [e for e in events 
                                  if e.get("type") in include_types 
                                  and e.get("confidence", 1.0) >= min_conf]
                        st.success(f"Would create highlight reel with {len(filtered)} clips")
                        st.json(filtered)
                
                elif compilation_type == "By Player":
                    players = list(set(e.get("player") for e in events if e.get("player")))
                    if players:
                        selected_player = st.selectbox("Select Player", players)
                        if st.button("Generate Player Compilation"):
                            player_events = [e for e in events if e.get("player") == selected_player]
                            st.success(f"Would create compilation for {selected_player} with {len(player_events)} clips")
                    else:
                        st.info("No player information available in events")
            else:
                st.info("No events available. Run an analysis first.")
        else:
            st.info("No analysis results yet. Analyze a video in the first tab.")
    
    # Footer
    st.divider()
    st.caption("Floorball LLM Analysis Tool | Built with Streamlit")


if __name__ == "__main__":
    main()
