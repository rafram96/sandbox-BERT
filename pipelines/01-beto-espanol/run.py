import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

subprocess.run(
    [
        sys.executable, "-m", "src.training.beto_experiments",
    ],
    cwd=ROOT,
    check=True,
)
