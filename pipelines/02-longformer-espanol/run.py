import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

subprocess.run(
    [
        sys.executable, "-m", "src.training.finetune",
        "--train", "data/espanol/resumes_kb_es_full.jsonl",
        "--test", "data/espanol/resumes_test_es_full.jsonl",
        "--model", "mrm8488/longformer-base-4096-spanish",
        "--max-len", "4096",
        "--batch", "2",
        "--epochs", "12",
        "--patience", "3",
        "--out", "output/longformer/model",
    ],
    cwd=ROOT,
    check=True,
)
