# Longformer — español y documentos largos

Entrena `mrm8488/longformer-base-4096-spanish` como clasificador.

Junta las dos cosas que hasta ahora estaban separadas: está entrenado en español
y además lee documentos largos, hasta unas 10 páginas de una sola vez. Los otros
modelos obligaban a elegir entre una cosa y la otra.

## Requisitos

Al leer documentos tan largos pide bastante más memoria de GPU que los demás
pipelines. Si se queda sin memoria, bajar `--batch` dentro de `run.py` (está en 2;
puede ir a 1). También se puede bajar `--max-len 4096` a 2048 para leer la mitad
de páginas y gastar menos memoria.

## Correr

```powershell
python pipelines/02-longformer-espanol/run.py
```

## Resultado

Pendiente de correr.

| | |
|---|---|
| Acierto | — |
| Tiempo | — |
| Datos | 6.195 entrenamiento / 2.681 prueba, 43 categorías |

Se espera que tarde bastante más que los pipelines de ~1.5 páginas, porque cada
documento es varias veces más largo.

## Usarlo en el sandbox

```
CLASSIFIER_MODE=finetuned
FT_MODEL_PATH=output/longformer/model
FT_TOKENIZER=mrm8488/longformer-base-4096-spanish
CONFIDENCE_METRIC=softmax
CONFIDENCE_THRESHOLD=0.90
```
