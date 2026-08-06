import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

subprocess.run(
    [
        sys.executable, "-m", "src.training.finetune",
        "--train", "resumes_kb_es.jsonl",
        "--test", "resumes_test_es.jsonl",
        "--model", "dccuchile/bert-base-spanish-wwm-cased",
        "--max-len", "256",
        "--batch", "16",
        "--epochs", "12",
        "--patience", "3",
        "--out", "ft-beto",
    ],
    cwd=ROOT,
    check=True,
)
