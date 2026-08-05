"""Embebe el corpus y recarga kb_documentos. Las categorias salen del corpus.

python -m src.corpus.ingest [--corpus RUTA.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import config
from ..core import db, embeddings


def cargar_corpus(path: Path):
    filas = []
    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea:
            filas.append(json.loads(linea))
    return filas


def _asegurar_categorias(cur, labels) -> dict:
    """Crea en 'categorias' las que falten y devuelve {codigo: id}."""
    cur.execute("SELECT codigo, id FROM categorias")
    cat_id = {codigo: cid for codigo, cid in cur}
    for lbl in sorted(labels):
        if lbl not in cat_id:
            cur.execute(
                "INSERT INTO categorias (codigo, descripcion) VALUES (:c, :d)",
                {"c": lbl, "d": f"Categoria {lbl}"},
            )
    cur.execute("SELECT codigo, id FROM categorias")
    return {codigo: cid for codigo, cid in cur}


def main(corpus_path: Path = None) -> None:
    corpus_path = corpus_path or config.KB_CORPUS
    print(f"== Ingesta: embeddings del corpus {corpus_path.name} ==")
    filas = cargar_corpus(corpus_path)
    textos = [f["texto"] for f in filas]

    print(f"  Backend de embeddings: {embeddings.backend_name()}  ({len(textos)} docs)")
    vecs = embeddings.embed(textos)
    if vecs.shape[1] != config.EMBEDDING_DIM:
        raise SystemExit(
            f"Dimension {vecs.shape[1]} != EMBEDDING_DIM {config.EMBEDDING_DIM}. "
            f"Ajusta EMBEDDING_DIM y el tipo VECTOR del esquema."
        )

    con = db.connect()
    try:
        cur = con.cursor()
        cat_id = _asegurar_categorias(cur, {f["categoria"] for f in filas})

        cur.execute("DELETE FROM kb_documentos")
        for fila, vec in zip(filas, vecs):
            cid = cat_id.get(fila["categoria"])
            if cid is None:
                raise SystemExit(f"Categoria desconocida en corpus: {fila['categoria']}")
            cur.execute(
                "INSERT INTO kb_documentos (texto, categoria_id, embedding) "
                "VALUES (:texto, :cid, :emb)",
                {"texto": fila["texto"], "cid": cid, "emb": db.to_vector(vec)},
            )
        con.commit()
        cur.execute("SELECT COUNT(*) FROM kb_documentos")
        print(f"  OK. Vectores cargados: {cur.fetchone()[0]}  |  categorias: {len(cat_id)}")
    finally:
        con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=None, help="Ruta a un .jsonl de corpus")
    a = ap.parse_args()
    main(Path(a.corpus) if a.corpus else None)
