# ModernBERT — documentos largos

Entrena `answerdotai/ModernBERT-base` como clasificador.

Es la opción para documentos extensos: procesa hasta ~20 páginas de una sola vez,
mientras que los modelos anteriores se quedan en ~1.5 páginas. Si los documentos
de mesa de partes tienen 10 o más páginas, esta arquitectura los lee completos.

## Requisitos

Necesita una GPU reciente (RTX 30xx en adelante). En GPUs anteriores funciona,
pero unas 20 veces más lento: en nuestras pruebas tardó 2h 10min contra los 6 min
de XLM-RoBERTa.

## Correr

```
python pipelines/01-modernbert-doc-largo/run.py
```

Dentro de `run.py` se pueden ajustar:

- `--max-len 4096` — subir a 8192 si la GPU tiene 24 GB o más.
- `--batch 4` — bajarlo si se queda sin memoria, subirlo si sobra.

## Resultado

| | |
|---|---|
| Acierto | 54.2% |
| Tiempo | 2h 10min |
| Datos | 645 documentos en español |

Este número no refleja el potencial del modelo: se corrió en una GPU antigua, con
pocos datos y sin usar su capacidad de documentos largos. Queda pendiente
repetirlo en una GPU adecuada con el corpus completo.

## Limitación

ModernBERT está entrenado solo en inglés, así que con textos en español parte en
desventaja frente a un modelo multilingüe. Ver alternativas en
[../README.md](../README.md).
