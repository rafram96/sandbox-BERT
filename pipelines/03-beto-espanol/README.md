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

## Evaluación completa

La evaluación actual usa los JSON completos de `data/espanol/` y reserva el
test exclusivamente para la medición final. Del corpus de entrenamiento separa
10% de validación de forma estratificada.

Primero se puede validar la preparación sin entrenar:

```
py -m src.training.beto_experiments --prepare-only
```

Para ejecutar todos los experimentos BETO:

```
py -m src.training.beto_experiments
```

También funciona el lanzador corto:

```
py pipelines/03-beto-espanol/run.py
```

Se entrenan dos modelos:

1. `direct`: primeros 512 tokens de cada documento.
2. `chunked`: todos los fragmentos de 512 tokens, con solape de 64.

El modelo chunked se evalúa una sola vez y genera cinco comparaciones:

- primer fragmento;
- todos + Mean Pooling;
- todos + Max Pooling;
- inicio/medio/final + Mean Pooling;
- inicio/medio/final + Max Pooling.

Cada variante reporta métricas para todo el test, documentos cortos y documentos
largos. Los modelos, checkpoints, predicciones, matrices de confusión y resúmenes
se guardan bajo `output/beto/`. Si una ejecución se interrumpe, el comando normal
reanuda el último checkpoint; si el modelo final ya existe, lo reutiliza.

Con los JSON completos verificados, la partición preparada es:

| Conjunto | Documentos |
|---|---:|
| Ajuste | 5,574 |
| Validación | 621 |
| Test corto (cabe en 512 tokens) | 1,097 |
| Test largo (requiere chunks) | 1,584 |

Los datos de entrenamiento producen 12,410 chunks y los de test 5,200.

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
