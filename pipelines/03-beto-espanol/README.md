# BETO — español

Entrena `dccuchile/bert-base-spanish-wwm-cased` como clasificador.

BETO es un modelo de la Universidad de Chile entrenado desde cero solo con textos
en español. La idea es compararlo contra XLM-RoBERTa, que sabe 100 idiomas a la
vez: uno se especializa en español, el otro reparte su capacidad entre muchos
idiomas. Cuál gana en nuestros documentos es lo que queremos medir.

Corre en cualquier GPU.

## Limitación

Solo procesa las primeras ~1.5 páginas de un documento; el resto lo descarta, el
mismo tope que XLM-RoBERTa.

## Correr

```
python pipelines/03-beto-espanol/run.py
```

## Resultado

Pendiente de correr.

| | |
|---|---|
| Acierto | — |
| Tiempo | — |
| Datos | 6195 entrenamiento / 2681 prueba, 43 categorías |

## Usarlo en el sandbox

```
CLASSIFIER_MODE=finetuned
FT_MODEL_PATH=ft-beto
FT_TOKENIZER=dccuchile/bert-base-spanish-wwm-cased
CONFIDENCE_METRIC=softmax
CONFIDENCE_THRESHOLD=0.90
```
