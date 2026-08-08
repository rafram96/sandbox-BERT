from __future__ import annotations

import argparse
import json

from .. import config
from ..core import db, embeddings
from ..corpus import bootstrap, ingest
from ..core.pipeline import Pipeline, Resultado


def _fmt(res: Resultado) -> str:
    cab = f"[{res.expediente or '-'}]  doc_id={res.documento_id}"
    base = f"  clasificador ligero -> {res.categoria_base}  (confianza {res.score_base:.2f})"
    if res.ruta == "rapida":
        via = f"  RUTA RAPIDA (>= {config.CONFIDENCE_THRESHOLD})  -> etiqueta final: {res.etiqueta_final}"
        gov = "  escalo_llm=No   revision_humana=No"
        cuerpo = "\n".join([cab, base, via, gov])
    else:
        via = f"  ESCALA A LLM (< {config.CONFIDENCE_THRESHOLD})"
        vecinos = "  RAG Top-K: " + ", ".join(
            f"{v.categoria}({v.similitud:.2f})" for v in res.vecinos
        )
        dec = f"  LLM -> etiqueta final: {res.etiqueta_final}"
        just = f"    {res.justificacion_llm}"
        gov = "  escalo_llm=Si   revision_humana=Si (regla 6.3)"
        cuerpo = "\n".join([cab, base, via, vecinos, dec, just, gov])
    return cuerpo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-setup", action="store_true")
    ap.add_argument("--umbral", type=float, default=None)
    args = ap.parse_args()

    print("== Esperando a Oracle 23ai ==")
    db.wait_until_ready(timeout=240)
    print("  BD lista.\n")

    if not args.skip_setup:
        bootstrap.main()
        print()
        ingest.main()
        print()
    else:
        print(f"(--skip-setup) usando esquema/embeddings existentes. "
              f"Backend: {embeddings.backend_name()}\n")

    umbral = args.umbral if args.umbral is not None else config.CONFIDENCE_THRESHOLD
    print(f"== Flujo de clasificacion (umbral de confianza = {umbral}) ==\n")
    pipe = Pipeline(umbral=umbral)

    entrantes = [
        json.loads(l) for l in config.ENTRANTES.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    n_rapida = n_llm = 0
    for doc in entrantes:
        res = pipe.procesar(doc["texto"], expediente=doc.get("expediente"))
        print(_fmt(res))
        print("-" * 78)
        n_rapida += res.ruta == "rapida"
        n_llm += res.ruta == "llm"

    print(f"\nResumen: {len(entrantes)} documentos  |  ruta rapida: {n_rapida}  |  escalaron a LLM: {n_llm}")
    print("Consulta los resultados persistidos con:  python -m src.scripts.query_resultados")


if __name__ == "__main__":
    main()
