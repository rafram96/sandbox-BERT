# Pipelines de evaluación

El repositorio conserva únicamente los dos modelos solicitados para la
comparación final.

| Pipeline | Uso principal | Longitud | Estrategias |
|---|---|---:|---|
| [BETO](01-beto-espanol/) | Español, documentos cortos | 512 tokens por entrada | directo, chunks, Mean/Max Pooling |
| [Longformer](02-longformer-espanol/) | Español, documentos largos | hasta 4.096 tokens | lectura directa del documento |

Ambos usan exactamente los mismos corpus completos de `data/espanol/`. BETO ya
separa validación desde el corpus de entrenamiento y reserva los 2.681 documentos
de test para la medición final. Longformer se alineará con ese mismo protocolo
antes de ejecutar su evaluación definitiva.

Todos los resultados deben guardarse bajo `output/`:

```text
output/
├── beto/
└── longformer/
```
