"""Simple CLI to run analysis on a video (simulated).

Usage: python scripts/run_analysis.py --video path\to\video.mp4
"""
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis import Analyzer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="clips")
    args = p.parse_args()

    analyzer = Analyzer(backend_name="simulated")
    res = analyzer.analyze_video(args.video, args.out)
    print("Events:")
    for ev in res["events"]:
        print(f" - {ev}")
    print("Clips:")
    for c in res["clips"]:
        print(f" - {c}")


if __name__ == "__main__":
    main()
