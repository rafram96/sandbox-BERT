# BETO — español

Entrena `dccuchile/bert-base-spanish-wwm-cased` como clasificador.

BETO es un modelo de la Universidad de Chile entrenado desde cero con textos en
español. Aquí se evalúa tanto con lectura directa como con chunking para medir su
comportamiento en documentos cortos y largos.

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
py pipelines/01-beto-espanol/run.py
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

La medición completa está pendiente. Los resultados válidos se escribirán en
`output/beto/`; no se conservan pilotos parciales dentro del repositorio.

## Usarlo en el sandbox

```
CLASSIFIER_MODE=finetuned
FT_MODEL_PATH=output/beto/models/direct
FT_TOKENIZER=dccuchile/bert-base-spanish-wwm-cased
CONFIDENCE_METRIC=softmax
CONFIDENCE_THRESHOLD=0.90
```
