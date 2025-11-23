"""Demo script showcasing all features of the Floorball LLM Analysis system."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_manager import AppConfig, SPORT_PRESETS
from src.analysis_enhanced import Analyzer
from src.llm_backends_enhanced import SimulatedLLM
from src.clip_manager import ClipManager
from src.cache import LLMCache


def demo_backends():
    """Demonstrate different backend options."""
    print("\n" + "="*60)
    print("DEMO: Available LLM Backends")
    print("="*60)
    
    backends = {
        "Simulated": "Offline, deterministic, no cost",
        "OpenAI GPT-4o": "Fast, structured outputs, $0.75/1M tokens",
        "Anthropic Claude": "High accuracy, $9/1M tokens",
        "Hugging Face": "Open models, ~$0.20/1M tokens",
        "Ollama": "Self-hosted, no per-request cost"
    }
    
    for name, desc in backends.items():
        print(f"  ✓ {name}: {desc}")


def demo_sport_presets():
    """Demonstrate sport-specific configurations."""
    print("\n" + "="*60)
    print("DEMO: Sport Presets")
    print("="*60)
    
    for sport_name, preset in SPORT_PRESETS.items():
        print(f"\n  {sport_name.upper()}:")
        print(f"    Events: {', '.join(preset.event_types[:5])}...")
        print(f"    Clip Padding: {preset.clip_padding_before}s before, {preset.clip_padding_after}s after")
        print(f"    Dedup Window: {preset.time_dedup_window}s")


def demo_configuration():
    """Demonstrate configuration management."""
    print("\n" + "="*60)
    print("DEMO: Configuration Management")
    print("="*60)
    
    config = AppConfig(
        llm_backend="simulated",
        sport="floorball",
        cache_enabled=True,
        log_level="INFO"
    )
    
    print("\n  Configuration created:")
    print(f"    Backend: {config.llm_backend}")
    print(f"    Sport: {config.sport}")
    print(f"    Cache: {'enabled' if config.cache_enabled else 'disabled'}")
    print(f"    Log Level: {config.log_level}")
    
    # Show sport preset
    preset = config.get_sport_preset()
    print(f"\n  Using {preset.name} preset:")
    print(f"    Event types: {len(preset.event_types)}")
    print(f"    Keywords: {len(preset.keywords)} categories")


def demo_caching():
    """Demonstrate caching functionality."""
    print("\n" + "="*60)
    print("DEMO: Response Caching")
    print("="*60)
    
    cache = LLMCache(cache_dir=".demo_cache", enabled=True)
    
    # Simulate cache operations
    test_response = {
        "events": [{"type": "goal", "timestamp": 100, "description": "Test goal"}],
        "meta": {"model": "test"}
    }
    
    cache.set("simulated", "test-model", "test input", test_response)
    retrieved = cache.get("simulated", "test-model", "test input")
    
    print(f"\n  Cache Test:")
    print(f"    Stored: {test_response}")
    print(f"    Retrieved: {retrieved}")
    print(f"    Match: {retrieved == test_response}")
    
    stats = cache.get_stats()
    print(f"\n  Cache Stats:")
    print(f"    Enabled: {stats.get('enabled', False)}")
    print(f"    Items: {stats.get('size', 0)}")
    
    # Cleanup
    cache.clear()


def demo_analysis():
    """Demonstrate video analysis workflow."""
    print("\n" + "="*60)
    print("DEMO: Video Analysis Workflow")
    print("="*60)
    
    config = AppConfig(llm_backend="simulated", sport="floorball")
    analyzer = Analyzer(config=config)
    
    # Health check
    health = analyzer.health_check()
    print(f"\n  Backend Health Check:")
    print(f"    Backend: {health['backend']}")
    print(f"    Status: {'✓ Healthy' if health['backend_healthy'] else '✗ Unhealthy'}")
    print(f"    Cache: {'enabled' if health['cache_enabled'] else 'disabled'}")
    
    # Simulate commentary analysis
    backend = SimulatedLLM()
    sample_text = """0:00 Match starts
0:45 Shot on target by player #7
1:20 Goal! Player #10 scores
2:10 Penalty called on player #5
3:30 Great save by the goalkeeper
"""
    
    result = backend.analyze_commentary(sample_text)
    print(f"\n  Analysis Results:")
    print(f"    Events Found: {len(result['events'])}")
    print(f"    Processing Time: {result['meta']['processing_ms']}ms")
    print(f"    Cost: ${result['meta']['cost_usd']:.6f}")
    
    print(f"\n  Detected Events:")
    for event in result['events']:
        print(f"    - {event['type'].upper()} at {event['timestamp']}s (conf: {event['confidence']})")


def demo_clip_management():
    """Demonstrate smart clipping features."""
    print("\n" + "="*60)
    print("DEMO: Smart Clipping & Compilations")
    print("="*60)
    
    manager = ClipManager()
    
    print("\n  Clip Manager Features:")
    print("    ✓ Event-specific padding (e.g., goals get more replay time)")
    print("    ✓ Highlight reel generation with filters")
    print("    ✓ Player-specific compilations")
    print("    ✓ Team-specific compilations")
    print("    ✓ Duration constraints (min/max)")
    
    # Show event-specific configs
    print("\n  Event-Specific Configurations:")
    for event_type in ["goal", "penalty", "shot"]:
        cfg = manager.get_clip_config_for_event(event_type)
        print(f"    {event_type}: {cfg.padding_before}s before, {cfg.padding_after}s after")


def demo_benchmarking():
    """Demonstrate benchmarking capabilities."""
    print("\n" + "="*60)
    print("DEMO: Benchmarking & Cost Analysis")
    print("="*60)
    
    print("\n  Benchmark Features:")
    print("    ✓ Multi-backend comparison")
    print("    ✓ Cost per request tracking")
    print("    ✓ Latency measurements")
    print("    ✓ Token usage tracking")
    print("    ✓ CSV/JSON export")
    print("    ✓ Statistical reports (min/max/avg)")
    
    print("\n  Metrics Tracked:")
    metrics = [
        "elapsed_ms", "processing_ms", "cost_usd",
        "events_found", "confidence_avg",
        "input_tokens", "output_tokens"
    ]
    for metric in metrics:
        print(f"    - {metric}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("FLOORBALL LLM ANALYSIS - FEATURE DEMONSTRATION")
    print("="*60)
    print("\nThis demo showcases all implemented features of the system.")
    
    try:
        demo_backends()
        demo_sport_presets()
        demo_configuration()
        demo_caching()
        demo_analysis()
        demo_clip_management()
        demo_benchmarking()
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Web UI:")
        print("   streamlit run scripts/web_ui.py")
        print("\n2. Run Benchmark:")
        print("   python scripts/benchmark_enhanced.py")
        print("\n3. Run Tests:")
        print("   pytest tests/test_enhanced.py -v")
        print("\n4. Configure for Production:")
        print("   cp config.yaml.example config.yaml")
        print("   # Edit config.yaml with your API keys")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
