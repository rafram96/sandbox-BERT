import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

subprocess.run(
    [
        sys.executable, "-m", "src.training.finetune",
        "--train", "resumes_kb_es.jsonl",
        "--test", "resumes_test_es.jsonl",
        "--model", "mrm8488/longformer-base-4096-spanish",
        "--max-len", "4096",
        "--batch", "2",
        "--epochs", "12",
        "--patience", "3",
        "--out", "ft-longformer",
    ],
    cwd=ROOT,
    check=True,
)
