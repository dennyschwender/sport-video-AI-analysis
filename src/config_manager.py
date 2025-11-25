import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class SportPreset:
    """Predefined configurations for different sports."""
    name: str
    event_types: List[str]
    clip_padding_before: int = 5  # seconds
    clip_padding_after: int = 5
    time_dedup_window: float = 5.0
    frame_interval: float = 10.0  # seconds between frames sent to LLM
    max_frames: int = 20  # maximum frames per API call
    hint: str = ""  # Sport-specific guidance for AI vision models
    keywords: Dict[str, List[str]] = field(default_factory=dict)


# Sport-specific presets (defaults - can be overridden in config.yaml)
DEFAULT_SPORT_PRESETS = {
    "floorball": SportPreset(
        name="floorball",
        event_types=["goal", "assist", "shot", "save", "penalty", "timeout"],
        clip_padding_before=10,
        clip_padding_after=5,
        frame_interval=0.5,  # Fast-paced sport, sample every 0.5 seconds
        max_frames=50,  # More frames for better accuracy
        hint="Look for: ball crossing the goal line, net wobble, referee confirming goal, teammates celebrating near the crease, and goalkeeper reactions. Avoid scoreboard-only changes.",
        keywords={
            "goal": ["goal", "score", "scores"],
            "assist": ["assist", "pass", "setup"],
            "shot": ["shot", "shoots", "attempt"],
            "save": ["save", "saves", "stopped", "block"],
            "penalty": ["penalty", "foul", "2 minutes", "bench"],
            "timeout": ["timeout", "time out", "break"]
        }
    ),
    "hockey": SportPreset(
        name="hockey",
        event_types=["goal", "assist", "shot", "save", "penalty", "icing", "offside"],
        clip_padding_before=6,
        clip_padding_after=10,
        frame_interval=10.0,  # Standard sampling
        max_frames=20,
        hint="Look for: puck crossing goal line, red goal light, players celebrating, goalkeeper saves, shots on goal, penalty calls.",
        keywords={
            "goal": ["goal", "score"],
            "penalty": ["penalty", "2 minutes", "5 minutes"],
            "icing": ["icing"],
            "offside": ["offside"]
        }
    ),
    "soccer": SportPreset(
        name="soccer",
        event_types=["goal", "assist", "shot", "save", "corner", "freekick", "penalty", "yellow_card", "red_card"],
        clip_padding_before=8,
        clip_padding_after=12,
        frame_interval=15.0,  # Slower sport, sample less frequently
        max_frames=15,  # Fewer frames needed
        hint="Look for: ball crossing goal line, players celebrating, goalkeeper saves, shots on goal, yellow/red cards, corner kicks.",
        keywords={
            "goal": ["goal", "score"],
            "corner": ["corner", "corner kick"],
            "yellow_card": ["yellow card", "booking"],
            "red_card": ["red card", "sent off"]
        }
    )
}

# Global SPORT_PRESETS (loaded from YAML or defaults)
SPORT_PRESETS = DEFAULT_SPORT_PRESETS.copy()


@dataclass
class AppConfig:
    """Main application configuration."""
    # Backend settings
    llm_backend: str = "simulated"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    huggingface_api_key: Optional[str] = None
    huggingface_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    ollama_endpoint: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    perplexity_api_key: Optional[str] = None
    perplexity_model: str = "sonar"
    
    # Processing settings
    sport: str = "floorball"
    clip_output_dir: str = "clips"
    cache_enabled: bool = True
    cache_dir: str = ".cache"
    parallel_processing: bool = False
    max_workers: int = 4
    
    # Rate limit settings (for chunk parallel processing)
    # OpenAI has lower rate limits (30K TPM for gpt-4o), use fewer workers
    max_workers_openai: Optional[int] = None
    # Gemini has higher rate limits, can handle more workers
    max_workers_gemini: int = 4
    # Perplexity rate limits, moderate parallelism
    max_workers_perplexity: int = 4
    # Delay between retries when hitting rate limits (seconds)
    rate_limit_retry_delay: float = 40.0
    # Maximum retries for rate-limited requests
    rate_limit_max_retries: int = 3
    
    # Rate limit values (used for automatic delay calculation)
    openai_rate_limit_tpm: int = 200000  # Tokens per minute
    openai_rate_limit_rpm: int = 500     # Requests per minute
    anthropic_rate_limit_tpm: int = 80000
    anthropic_rate_limit_rpm: int = 50
    
    # Retry settings (general API errors)
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/floorball_llm.log"
    # Goal confirmation and annotation enhancements
    goal_refinement_enabled: bool = True
    goal_refinement_attempts: int = 2
    goal_refinement_window: float = 2.5
    goal_refinement_interval: float = 0.25
    goal_annotation_enabled: bool = False
    goal_annotation_dir: str = "annotations/goals"
    goal_annotation_threshold: float = 0.7
    
    def get_sport_preset(self) -> SportPreset:
        """Get the preset for the configured sport."""
        return SPORT_PRESETS.get(self.sport, SPORT_PRESETS["floorball"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Load sport presets if present in YAML
        global SPORT_PRESETS
        if data and 'sport_presets' in data:
            SPORT_PRESETS.clear()
            for sport_name, preset_data in data['sport_presets'].items():
                SPORT_PRESETS[sport_name] = SportPreset(
                    name=preset_data.get('name', sport_name),
                    event_types=preset_data.get('event_types', []),
                    clip_padding_before=preset_data.get('clip_padding_before', 5),
                    clip_padding_after=preset_data.get('clip_padding_after', 5),
                    time_dedup_window=preset_data.get('time_dedup_window', 5.0),
                    frame_interval=preset_data.get('frame_interval', 10.0),
                    max_frames=preset_data.get('max_frames', 20),
                    hint=preset_data.get('hint', ''),
                    keywords=preset_data.get('keywords', {})
                )
            # Remove sport_presets from data before creating config
            del data['sport_presets']
        
        return cls.from_dict(data or {})
    
    def to_yaml(self, path: Path):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Override from environment
        if os.getenv("LLM_BACKEND"):
            config.llm_backend = os.getenv("LLM_BACKEND")
        if os.getenv("OPENAI_API_KEY"):
            config.openai_api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY"):
            config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if os.getenv("HUGGINGFACE_API_KEY"):
            config.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if os.getenv("SPORT"):
            config.sport = os.getenv("SPORT")
        if os.getenv("CACHE_ENABLED"):
            config.cache_enabled = os.getenv("CACHE_ENABLED").lower() == "true"
        
        return config


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration with smart separation:
    
    Priority:
    1. config.yaml (or provided path) - Application settings (models, sport, etc.)
    2. .env file - Secrets/API keys (loaded via python-dotenv)
    3. Defaults - Built-in sensible defaults
    
    API keys from .env ALWAYS override config.yaml (for security).
    """
    # Try to load from config.yaml first (app settings)
    if config_path is None:
        config_path = Path("config.yaml")
    
    if config_path.exists():
        # Load app settings from YAML
        config = AppConfig.from_yaml(config_path)
    else:
        # Use defaults if no config.yaml
        config = AppConfig()
    
    # Always merge API keys from environment variables (.env file)
    # This ensures secrets are never in config.yaml
    if os.getenv("OPENAI_API_KEY"):
        config.openai_api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("ANTHROPIC_API_KEY"):
        config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if os.getenv("HUGGINGFACE_API_KEY"):
        config.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if os.getenv("GEMINI_API_KEY"):
        config.gemini_api_key = os.getenv("GEMINI_API_KEY")
    if os.getenv("PERPLEXITY_API_KEY"):
        config.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    
    # Allow environment override for deployment (optional)
    if os.getenv("LLM_BACKEND"):
        config.llm_backend = os.getenv("LLM_BACKEND")
    if os.getenv("SPORT"):
        config.sport = os.getenv("SPORT")
    
    return config
