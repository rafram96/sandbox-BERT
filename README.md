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
py -m src.scripts.run_demo
py -m src.scripts.query_resultados
```

Dataset de resumes (Kaggle resume-data-pdf, PDFs escaneados -> OCR):

```bash
py -m src.corpus.load_resumes --per-cat 22        # OCR + split train/test
py -m src.corpus.ingest --corpus data/resumes_kb.jsonl
py -m src.scripts.evaluate --test data/resumes_test.jsonl
```

## Pipelines de clasificación

Dos opciones de entrenamiento, cada una en su carpeta:

| | Documentos | Idiomas | Acierto |
|---|---|---|---|
| [01 — ModernBERT](pipelines/01-modernbert-doc-largo/) | hasta ~20 páginas | inglés | 54.2% |
| [02 — XLM-RoBERTa](pipelines/02-xlmroberta-multilingue/) | hasta ~1.5 páginas | multilingüe | 80.5% |

En [pipelines/README.md](pipelines/README.md) está la comparación y por qué
XLM-RoBERTa no sirve para documentos largos.

## Config (`.env`)

- `EMBEDDING_MODEL` — modelo de embeddings (nomic-ai/modernbert-embed-base)
- `CONFIDENCE_METRIC` / `CONFIDENCE_THRESHOLD` — puerta de confianza (margen, 0.05)
- `OLLAMA_MODEL` — LLM del escalado; `USE_MOCK_LLM=1` para no usar Ollama
- `FORCE_HASH_EMBEDDING=1` — corre sin torch (embedding hash, solo infra)

## Estructura

```
sql/               esquema (tabla VECTOR 768) y seed
data/              corpus y sets de prueba (.jsonl)
pipelines/         opciones de entrenamiento (ModernBERT / XLM-RoBERTa)
src/
  config.py        configuración (.env)
  core/            db, embeddings, classifier, rag, llm, pipeline
  corpus/          bootstrap, ingest, load_resumes, translate
  training/        finetune
  scripts/         run_demo, evaluate, calibrar, query_resultados, run_es_pipeline
```
