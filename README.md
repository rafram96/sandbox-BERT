# SUN-Sandbox

Clasificación de documentos sobre Oracle 23ai. Un clasificador por embeddings
resuelve los casos claros y escala a RAG + LLM los ambiguos.

Flujo: `embedding → clasificador (centroides) → margen ≥ umbral ? ruta rápida : RAG Top-K + LLM`

## Requisitos

- Docker
- Python 3.10+ (`py` en Windows)
- Ollama con un modelo instalado (para el paso LLM)

## Uso

```bash
docker compose up -d          # Oracle 23ai (esperar healthy)
py -m pip install -r requirements.txt
cp .env.example .env
```

Demo SUNAFIL:

```bash
py run_demo.py
py query_resultados.py
```

Dataset de resumes (Kaggle resume-data-pdf, PDFs escaneados -> OCR):

```bash
py -m src.load_resumes --per-cat 22           # OCR + split train/test
py -m src.ingest --corpus data/resumes_kb.jsonl
py evaluate.py --test data/resumes_test.jsonl
```

## Config (`.env`)

- `EMBEDDING_MODEL` — modelo de embeddings (nomic-ai/modernbert-embed-base)
- `CONFIDENCE_METRIC` / `CONFIDENCE_THRESHOLD` — puerta de confianza (margen, 0.05)
- `OLLAMA_MODEL` — LLM del escalado; `USE_MOCK_LLM=1` para no usar Ollama
- `FORCE_HASH_EMBEDDING=1` — corre sin torch (embedding hash, solo infra)

## Estructura

```
sql/            esquema (tabla VECTOR 768) y seed
data/           corpus y sets de prueba (.jsonl)
src/            db, embeddings, classifier, rag, llm, pipeline, ingest, load_resumes
run_demo.py     demo end-to-end
evaluate.py     métricas sobre un test etiquetado
```
