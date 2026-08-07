# Pipelines de clasificación

Cuatro opciones de entrenamiento, cada una en su carpeta.

| | [01 — ModernBERT](01-modernbert-doc-largo/) | [02 — XLM-RoBERTa](02-xlmroberta-multilingue/) | [03 — BETO](03-beto-espanol/) | [04 — Longformer](04-longformer-espanol/) |
|---|---|---|---|---|
| Tamaño de documento | Hasta ~20 páginas | Hasta ~1.5 páginas | Hasta ~1.5 páginas | Hasta ~10 páginas |
| Idiomas | Solo inglés | 100 idiomas, incluye español | Solo español | Solo español |
| GPU | Necesita una reciente (RTX 30xx en adelante) | Cualquiera | Cualquiera | Cualquiera, con bastante memoria |
| Acierto medido | 54.2% | 80.5% | piloto 4.0% (no concluyente) | pendiente |
| Tiempo | 2h 10min | 6 min | 15.1 min en CPU (1 época/645 docs) | pendiente |

El 54.2% de ModernBERT se midió en una GPU antigua, con pocos datos y sin
aprovechar su capacidad de documentos largos, así que no refleja su potencial.

## ¿XLM-RoBERTa sirve para documentos largos?

No. Solo procesa las primeras ~1.5 páginas de un documento; el resto lo descarta.
Es un límite fijo de su diseño (512 tokens), no algo que se pueda configurar.

ModernBERT sí: procesa hasta ~20 páginas, y esa es justamente su principal
ventaja frente a modelos anteriores.

Qué tan largos son nuestros documentos:

| Corpus | Tamaño típico | Se pasan del límite |
|---|---|---|
| Español (recortado) | ~1/2 página | 0% |
| Inglés (texto completo) | ~1.5 páginas | 76% |

El 80.5% se obtuvo con documentos que entraban completos, así que ese resultado
vale para documentos cortos. Con documentos de 10 o más páginas habría que
volver a medir: **esa prueba está pendiente**.

Si se necesitan documentos largos en español, hay tres caminos:

1. Usar ModernBERT (opción 01), aunque está entrenado en inglés.
2. Partir el documento en pedazos, clasificar cada uno y combinar el resultado.
   Reutiliza el modelo que ya da 80.5%. **Ya está implementado**:

   ```
   python -m src.training.chunking --test resumes_test_es.jsonl \
       --model-path ft-xlmroberta --tokenizer xlm-roberta-base
   ```

   Combina los pedazos promediando (`--agregacion mean`) o quedándose con el
   pedazo más seguro (`--agregacion max`), y con `--paginas extremos` mira solo la
   primera página, la del medio y la última, que suele bastar y es mucho más
   rápido. Al final imprime cuánto mejora frente a leer solo la primera parte.
3. Usar un modelo en español que acepte documentos largos: es el pipeline 04.

Para documentos de mesa de partes, la opción 2 suele ser la más práctica: en un
documento largo, lo que define su categoría casi nunca está repartido por todas
las páginas.

## Instalación

```
pip install torch transformers accelerate numpy
```

Con GPU hay que instalar torch con soporte CUDA, porque la versión por defecto en
Windows es solo para CPU:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"
```

Debe imprimir `True`.
