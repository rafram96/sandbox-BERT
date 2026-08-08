# SUN-Sandbox

Evaluación reproducible de modelos para clasificar currículums en español. El
repositorio compara dos enfoques:

- **BETO**, para documentos cortos y documentos largos mediante chunking.
- **Longformer en español**, para procesar documentos largos directamente.

## Datos oficiales

Solo hay cuatro corpus de evaluación:

```text
data/
├── ingles/
│   ├── resumes_kb_en_full.jsonl       # 6.195 documentos de entrenamiento
│   └── resumes_test_en_full.jsonl     # 2.681 documentos de test
└── espanol/
    ├── resumes_kb_es_full.jsonl       # traducción completa de entrenamiento
    └── resumes_test_es_full.jsonl     # traducción completa de test
```

Los archivos conservan las 43 categorías y el orden de los originales. Los
JSON pequeños usados por la demostración del sistema están aislados en
`data/demo/` y no participan en la evaluación de modelos.

## Instalación

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En Windows, PyTorch debe instalarse con la versión CUDA apropiada para la GPU.

## BETO

```powershell
python -m src.training.beto_experiments --prepare-only
python -m src.training.beto_experiments
```

El pipeline compara lectura directa contra chunking con Mean Pooling y Max
Pooling. Guarda modelos, predicciones, métricas y matrices de confusión en
`output/beto/`. Más detalles en
[`pipelines/01-beto-espanol/`](pipelines/01-beto-espanol/).

## Longformer

```powershell
python pipelines/02-longformer-espanol/run.py
```

Los artefactos se guardan en `output/longformer/`. Más detalles en
[`pipelines/02-longformer-espanol/`](pipelines/02-longformer-espanol/).

## Traducción reproducible

Las traducciones ya están incluidas. Para regenerarlas con MarianMT:

```powershell
python -m src.corpus.translate_mt --in data/ingles/resumes_kb_en_full.jsonl --out data/espanol/resumes_kb_es_full.jsonl
python -m src.corpus.translate_mt --in data/ingles/resumes_test_en_full.jsonl --out data/espanol/resumes_test_es_full.jsonl
```

Usa `--resume` para continuar una traducción interrumpida.

## Otras partes

- `src/training/`: entrenamiento y evaluación.
- `src/corpus/`: traducción completa e ingestión.
- `src/core/`, `src/scripts/`, `sql/`: demostración del clasificador integrado
  con Oracle 23ai.
- `visual-test.html`: panel local para pruebas visuales.
