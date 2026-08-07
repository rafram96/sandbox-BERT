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

### Piloto de infraestructura (CPU)

Se verificó el pipeline de punta a punta con el checkpoint oficial
`dccuchile/bert-base-spanish-wwm-cased` y una muestra balanceada de documentos
cortos del corpus principal.

| | |
|---|---|
| Acierto | 4.0% (12/301), **no concluyente** |
| Entrenamiento | 15.1 min, 1 época, CPU |
| Evaluación | 73 s |
| Datos | 645 entrenamiento / 301 prueba, 43 categorías |
| Longitud | 256 tokens |
| Checkpoint | `ft-beto-short-pilot/` |

El piloto usa solo 15 documentos de entrenamiento por categoría y una época. Su
objetivo es comprobar que la descarga, tokenización, entrenamiento, evaluación y
guardado funcionan; **no debe compararse** con el 80.5% de XLM-RoBERTa, obtenido
con 6195 documentos y cinco épocas.

Una medición comparable sigue pendiente y debe ejecutarse preferentemente en
GPU con el comando normal de este pipeline. Extrapolando el rendimiento medido
en esta máquina, las 12 épocas sobre el corpus completo tardarían alrededor de
30 horas en CPU.

### Auditoría de longitud

En el corpus principal, 92.1% de los documentos de entrenamiento y 92.3% de los
de prueba caben completos en 256 tokens. Por tanto, este corpus sí es adecuado
para medir BETO en documentos cortos. El subconjunto ubicado bajo `data/` no lo
es: más de 95% supera los 256 tokens y queda truncado.

## Usarlo en el sandbox

```
CLASSIFIER_MODE=finetuned
FT_MODEL_PATH=ft-beto
FT_TOKENIZER=dccuchile/bert-base-spanish-wwm-cased
CONFIDENCE_METRIC=softmax
CONFIDENCE_THRESHOLD=0.90
```
