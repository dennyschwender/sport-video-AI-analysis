from typing import Dict, Any, Optional
from pathlib import Path

from src.llm_backends_enhanced import (
    SimulatedLLM, HuggingFaceBackend, OllamaBackend
)
from src.video_tools import find_audio_transcript_for_video
from src.schema import LLMResponse
from src.clip_manager import ClipManager, ClipConfig
from src.cache import LLMCache
from src.logger import Logger, log_llm_call, log_error_with_context
from src.config_manager import AppConfig


class Analyzer:
    """Enhanced analyzer with caching, logging, and error handling."""
    
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        cache: Optional[LLMCache] = None,
        logger: Optional[Logger] = None
    ):
        self.config = config or AppConfig()
        self.cache = cache or LLMCache(
            cache_dir=self.config.cache_dir,
            enabled=self.config.cache_enabled
        )
        self.logger = (logger or Logger.get_logger(
            log_file=self.config.log_file,
            level=self.config.log_level
        ))
        
        # Initialize backend
        self.backend = self._init_backend()
        self.clip_manager = ClipManager(self._get_clip_config())
    
    def _init_backend(self):
        """Initialize the configured LLM backend."""
        backend_name = self.config.llm_backend
        
        self.logger.info(f"Initializing backend: {backend_name}")
        
        try:
            if backend_name == "simulated":
                return SimulatedLLM()
            
            elif backend_name == "openai":
                from src.llm_backends_enhanced import OpenAIBackend
                if not self.config.openai_api_key:
                    raise ValueError("OpenAI API key not configured")
                return OpenAIBackend(
                    api_key=self.config.openai_api_key,
                    model=self.config.openai_model,
                    max_retries=self.config.max_retries,
                    timeout=self.config.timeout
                )
            
            elif backend_name == "anthropic":
                from src.llm_backends_enhanced import AnthropicBackend
                if not self.config.anthropic_api_key:
                    raise ValueError("Anthropic API key not configured")
                return AnthropicBackend(
                    api_key=self.config.anthropic_api_key,
                    model=self.config.anthropic_model,
                    max_retries=self.config.max_retries,
                    timeout=self.config.timeout
                )
            
            elif backend_name == "gemini":
                from src.llm_backends_enhanced import GeminiBackend
                if not self.config.gemini_api_key:
                    raise ValueError("Gemini API key not configured")
                return GeminiBackend(
                    api_key=self.config.gemini_api_key,
                    model=self.config.gemini_model,
                    max_retries=self.config.max_retries,
                    timeout=self.config.timeout
                )
            
            elif backend_name == "huggingface":
                return HuggingFaceBackend(
                    api_key=self.config.huggingface_api_key,
                    model=self.config.huggingface_model,
                    max_retries=self.config.max_retries,
                    timeout=self.config.timeout
                )
            
            elif backend_name == "ollama":
                return OllamaBackend(
                    endpoint=self.config.ollama_endpoint,
                    model=self.config.ollama_model,
                    timeout=self.config.timeout
                )
            
            else:
                self.logger.warning(f"Unknown backend '{backend_name}', falling back to simulated")
                return SimulatedLLM()
        
        except Exception as e:
            self.logger.error(f"Failed to initialize backend '{backend_name}': {e}")
            self.logger.warning("Falling back to simulated backend")
            return SimulatedLLM()
    
    def _get_clip_config(self) -> ClipConfig:
        """Get clip configuration from sport preset."""
        preset = self.config.get_sport_preset()
        return ClipConfig(
            padding_before=preset.clip_padding_before,
            padding_after=preset.clip_padding_after
        )
    
    def analyze_video(
        self,
        video_path: str,
        out_dir: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Analyze video and extract events."""
        out_dir = out_dir or self.config.clip_output_dir
        
        self.logger.info(f"Starting analysis for video: {video_path}")
        
        # Find transcript
        transcript = find_audio_transcript_for_video(video_path)
        if not transcript:
            self.logger.warning("No transcript found for video")
            return {
                "events": [],
                "clips": [],
                "meta": {"error": "no_transcript"},
                "cache_hit": False
            }
        
        # Check cache
        cache_hit = False
        model_name = getattr(self.backend, 'model', 'unknown')
        
        if use_cache and self.cache.enabled:
            cached_response = self.cache.get(
                self.config.llm_backend,
                model_name,
                transcript
            )
            if cached_response:
                self.logger.info("Using cached LLM response")
                resp = cached_response
                cache_hit = True
        
        # Call LLM if not cached
        if not cache_hit:
            try:
                resp = self.backend.analyze_commentary(transcript)
                
                # Cache the response
                if use_cache and self.cache.enabled:
                    self.cache.set(
                        self.config.llm_backend,
                        model_name,
                        transcript,
                        resp
                    )
                
                # Log metrics
                meta = resp.get("meta", {})
                log_llm_call(
                    self.logger,
                    self.config.llm_backend,
                    model_name,
                    len(transcript),
                    meta.get("processing_ms", 0),
                    meta.get("cost_usd", 0.0),
                    len(resp.get("events", []))
                )
            
            except Exception as e:
                log_error_with_context(
                    self.logger,
                    e,
                    {"video_path": video_path, "backend": self.config.llm_backend}
                )
                return {
                    "events": [],
                    "clips": [],
                    "meta": {"error": str(e)},
                    "cache_hit": False
                }
        
        # Parse and deduplicate events
        try:
            parsed = LLMResponse(**resp)
            preset = self.config.get_sport_preset()
            events = parsed.deduplicate_events(time_window=preset.time_dedup_window)
            
            self.logger.info(f"Found {len(events)} events after deduplication")
            
            # Prepare clips
            clips = self.clip_manager.prepare_clips(events, video_path, out_dir)
            
            return {
                "events": [e.model_dump() for e in events],
                "clips": clips,
                "meta": resp.get("meta", {}),
                "cache_hit": cache_hit
            }
        
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"video_path": video_path, "stage": "event_parsing"}
            )
            # Fallback to raw events
            return {
                "events": resp.get("events", []),
                "clips": [],
                "meta": resp.get("meta", {}),
                "cache_hit": cache_hit
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of backend and dependencies."""
        backend_healthy = False
        try:
            backend_healthy = self.backend.health_check()
        except Exception as e:
            self.logger.error(f"Backend health check failed: {e}")
        
        return {
            "backend": self.config.llm_backend,
            "backend_healthy": backend_healthy,
            "cache_enabled": self.cache.enabled,
            "cache_stats": self.cache.get_stats()
        }
