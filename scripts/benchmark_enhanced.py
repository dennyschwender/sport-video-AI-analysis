"""Enhanced benchmarking with CSV logging and comparison reports."""
import time
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_backends_enhanced import (
    SimulatedLLM, HuggingFaceBackend, OllamaBackend
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    timestamp: str
    backend: str
    model: str
    input_length: int
    elapsed_ms: int
    processing_ms: Optional[int]
    cost_usd: float
    events_found: int
    success: bool
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    confidence_avg: Optional[float] = None


class BenchmarkRunner:
    """Run benchmarks across multiple backends and log results."""
    
    def __init__(self, log_dir: str = "benchmark_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(
        self,
        backends: List[Any],
        transcript: str,
        iterations: int = 1
    ) -> List[BenchmarkResult]:
        """Run benchmark across all backends."""
        all_results = []
        
        for backend in backends:
            for i in range(iterations):
                print(f"Running iteration {i+1}/{iterations} for {backend.__class__.__name__}...")
                result = self._benchmark_single(backend, transcript)
                all_results.append(result)
                self.results.append(result)
                
                # Small delay between iterations
                if i < iterations - 1:
                    time.sleep(0.5)
        
        return all_results
    
    def _benchmark_single(self, backend: Any, transcript: str) -> BenchmarkResult:
        """Benchmark a single backend."""
        backend_name = backend.__class__.__name__
        model_name = getattr(backend, 'model', 'unknown')
        
        start = time.time()
        success = True
        error = None
        
        try:
            result = backend.analyze_commentary(transcript)
            elapsed_ms = int((time.time() - start) * 1000)
            
            events = result.get("events", [])
            meta = result.get("meta", {})
            
            # Calculate average confidence
            confidences = [e.get("confidence", 0) for e in events if e.get("confidence")]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            
            return BenchmarkResult(
                timestamp=datetime.now().isoformat(),
                backend=backend_name,
                model=model_name,
                input_length=len(transcript),
                elapsed_ms=elapsed_ms,
                processing_ms=meta.get("processing_ms"),
                cost_usd=meta.get("cost_usd", 0.0),
                events_found=len(events),
                success=True,
                error=meta.get("error"),
                input_tokens=meta.get("input_tokens"),
                output_tokens=meta.get("output_tokens"),
                confidence_avg=avg_confidence
            )
        
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return BenchmarkResult(
                timestamp=datetime.now().isoformat(),
                backend=backend_name,
                model=model_name,
                input_length=len(transcript),
                elapsed_ms=elapsed_ms,
                processing_ms=None,
                cost_usd=0.0,
                events_found=0,
                success=False,
                error=str(e)
            )
    
    def save_to_csv(self, filename: Optional[str] = None):
        """Save benchmark results to CSV."""
        if not self.results:
            print("No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.csv"
        
        filepath = self.log_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))
        
        print(f"Saved {len(self.results)} results to {filepath}")
        return filepath
    
    def save_to_json(self, filename: Optional[str] = None):
        """Save benchmark results to JSON."""
        if not self.results:
            print("No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        print(f"Saved {len(self.results)} results to {filepath}")
        return filepath
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comparison report from results."""
        if not self.results:
            return {}
        
        # Group by backend
        by_backend: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            key = f"{result.backend}:{result.model}"
            if key not in by_backend:
                by_backend[key] = []
            by_backend[key].append(result)
        
        # Calculate statistics
        report = {}
        for key, results in by_backend.items():
            successful = [r for r in results if r.success]
            if not successful:
                continue
            
            report[key] = {
                "runs": len(results),
                "success_rate": len(successful) / len(results),
                "avg_latency_ms": sum(r.elapsed_ms for r in successful) / len(successful),
                "avg_cost_usd": sum(r.cost_usd for r in successful) / len(successful),
                "avg_events_found": sum(r.events_found for r in successful) / len(successful),
                "avg_confidence": sum(r.confidence_avg for r in successful if r.confidence_avg) / len([r for r in successful if r.confidence_avg]) if any(r.confidence_avg for r in successful) else None,
                "total_cost_usd": sum(r.cost_usd for r in successful),
                "min_latency_ms": min(r.elapsed_ms for r in successful),
                "max_latency_ms": max(r.elapsed_ms for r in successful)
            }
        
        return report
    
    def print_report(self):
        """Print a formatted comparison report."""
        report = self.generate_report()
        
        if not report:
            print("No successful results to report")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON REPORT")
        print("="*80)
        
        for backend, stats in report.items():
            print(f"\n{backend}")
            print("-" * 60)
            print(f"  Runs: {stats['runs']}")
            print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
            print(f"  Avg Latency: {stats['avg_latency_ms']:.0f}ms (min: {stats['min_latency_ms']}ms, max: {stats['max_latency_ms']}ms)")
            print(f"  Avg Cost: ${stats['avg_cost_usd']:.6f}")
            print(f"  Total Cost: ${stats['total_cost_usd']:.6f}")
            print(f"  Avg Events: {stats['avg_events_found']:.1f}")
            if stats['avg_confidence']:
                print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
        
        print("\n" + "="*80)


def main():
    """Run benchmark with available backends."""
    sample_transcript = """0:00 Match starts, kickoff by team A
0:45 Shot on goal by player #7, saved by goalkeeper
1:20 Goal! Player #10 scores for team A, assisted by #7
2:10 Penalty called on team B, player #5 to penalty box
3:30 Shot by player #12, goes wide
4:15 Goal! Player #22 scores for team B
5:00 Timeout called by team A
6:30 Great save by goalkeeper on shot from #10
7:45 Turnover, team B gains possession
8:20 Shot by #15, another save
9:00 Goal! Player #7 scores, hat trick!
"""
    
    # Initialize available backends
    backends = [
        SimulatedLLM(),
        HuggingFaceBackend(model="hf-simulated"),
        OllamaBackend()
    ]
    
    # Try to add OpenAI if available
    try:
        import os
        from src.llm_backends_enhanced import OpenAIBackend
        if os.getenv("OPENAI_API_KEY"):
            backends.append(OpenAIBackend(api_key=os.getenv("OPENAI_API_KEY")))
    except Exception:
        pass
    
    # Try to add Anthropic if available
    try:
        import os
        from src.llm_backends_enhanced import AnthropicBackend
        if os.getenv("ANTHROPIC_API_KEY"):
            backends.append(AnthropicBackend(api_key=os.getenv("ANTHROPIC_API_KEY")))
    except Exception:
        pass
    
    runner = BenchmarkRunner()
    
    print(f"Running benchmarks with {len(backends)} backends...")
    runner.run_benchmark(backends, sample_transcript, iterations=3)
    
    # Save results
    runner.save_to_csv()
    runner.save_to_json()
    
    # Print report
    runner.print_report()


if __name__ == '__main__':
    main()
