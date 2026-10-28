"""Trained metrics evaluation entrypoint."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).parent / "metrics_computation.py"
    runpy.run_path(str(target), run_name="__main__")