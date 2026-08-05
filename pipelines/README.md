# Pipelines de clasificación

Dos opciones de entrenamiento, cada una en su carpeta.

| | [01 — ModernBERT](01-modernbert-doc-largo/) | [02 — XLM-RoBERTa](02-xlmroberta-multilingue/) |
|---|---|---|
| Tamaño de documento | Hasta ~20 páginas | Hasta ~1.5 páginas |
| Idiomas | Solo inglés | 100 idiomas, incluye español |
| GPU | Necesita una reciente (RTX 30xx en adelante) | Cualquiera |
| Acierto medido | 54.2% | 80.5% |
| Tiempo | 2h 10min | 6 min |

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
   Reutiliza el modelo que ya da 80.5%.
3. Buscar un modelo multilingüe que acepte documentos largos.

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
