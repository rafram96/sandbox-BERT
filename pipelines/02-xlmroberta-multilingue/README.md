# XLM-RoBERTa — multilingüe

Entrena `xlm-roberta-base` como clasificador. Es el que mejor resultado dio hasta
ahora.

Está entrenado en 100 idiomas, incluido español, y corre en cualquier GPU.

## Limitación

Solo procesa las primeras ~1.5 páginas de un documento; el resto lo descarta.

| Corpus | Tamaño típico | Se pasan del límite |
|---|---|---|
| Español (recortado) | ~1/2 página | 0% |
| Inglés (texto completo) | ~1.5 páginas | 76% |

El 80.5% se midió con documentos que entraban completos, sin recortar. Con
documentos largos este modelo perdería contenido.

## Correr

```
python pipelines/02-xlmroberta-multilingue/run.py
```

## Resultado

| | |
|---|---|
| Acierto | 80.5% |
| Tiempo | 6 min (5 épocas) |
| Datos | 6195 entrenamiento / 2681 prueba, 43 categorías |

Con el umbral de confianza en 0.90, de los 2681 documentos de prueba:

- 1775 (66%) se resuelven directo, con 92% de acierto
- 906 (34%) se derivan al LLM

## Usarlo en el sandbox

```
CLASSIFIER_MODE=finetuned
FT_MODEL_PATH=ft-xlmroberta
FT_TOKENIZER=xlm-roberta-base
CONFIDENCE_METRIC=softmax
CONFIDENCE_THRESHOLD=0.90
```
