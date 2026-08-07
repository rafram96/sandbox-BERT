"""Configuracion central del sandbox. Lee variables de entorno (.env)."""
import os
from pathlib import Path

# Carga .env si existe (sin dependencia dura de python-dotenv).
_ROOT = Path(__file__).resolve().parent.parent
_ENV = _ROOT / ".env"
if _ENV.exists():
    for _line in _ENV.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

ROOT = _ROOT

################# DB
ORA_USER = os.environ.get("ORA_USER", "sunafil")
ORA_PASSWORD = os.environ.get("ORA_PASSWORD", "sunafil")
ORA_DSN = os.environ.get("ORA_DSN", "localhost:1521/FREEPDB1")

################# Flujo
# Clasificador: "centroide" (embeddings, sin entrenar) | "finetuned" (finetune.py)
CLASSIFIER_MODE = os.environ.get("CLASSIFIER_MODE", "centroide")
FT_MODEL_PATH = os.environ.get("FT_MODEL_PATH", "ft-model")
# Tokenizer del modelo fine-tuned; si el checkpoint no lo trae, se usa este (la base).
FT_TOKENIZER = os.environ.get("FT_TOKENIZER", "xlm-roberta-base")

# Puerta de confianza: "margen" (top1-top2, para centroides) o "softmax" (para finetuned)
CONFIDENCE_METRIC = os.environ.get("CONFIDENCE_METRIC", "margen")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.05"))
TOP_K = int(os.environ.get("TOP_K", "5"))

################# Embeddings / ModernBERT
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "answerdotai/ModernBERT-base")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))
FORCE_HASH_EMBEDDING = os.environ.get("FORCE_HASH_EMBEDDING", "0") == "1"
# Prefijos doc/consulta de modelos tipo nomic-embed, sin espacio final
EMBED_DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "")
EMBED_QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "")

################# LLM local (Ollama) para el paso de escalado
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M")
USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "0") == "1"

################# Rutas de datos
KB_CORPUS = ROOT / "data" / "kb_corpus.jsonl"
ENTRANTES = ROOT / "data" / "entrantes.jsonl"
SQL_DIR = ROOT / "sql"
