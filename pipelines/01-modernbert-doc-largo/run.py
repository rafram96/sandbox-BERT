import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

subprocess.run(
    [
        sys.executable, "-m", "src.training.finetune",
        "--train", "data/resumes_kb.jsonl",
        "--test", "data/resumes_test.jsonl",
        "--model", "answerdotai/ModernBERT-base",
        "--max-len", "4096",
        "--batch", "4",
        "--epochs", "12",
        "--patience", "3",
        "--out", "ft-modernbert",
    ],
    cwd=ROOT,
    check=True,
)
