"""Recuperacion Top-K con VECTOR_DISTANCE COSINE sobre Oracle 23ai."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from . import db, embeddings


@dataclass
class Vecino:
    kb_id: int
    categoria: str
    distancia: float          # 0 = identico, 2 = opuesto (coseno)
    texto: str

    @property
    def similitud(self) -> float:
        return 1.0 - self.distancia


def top_k(texto: str, k: int = 5) -> List[Vecino]:
    qvec = db.to_vector(embeddings.embed_one(texto, is_query=True))
    con = db.connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT k.id,
                   c.codigo,
                   VECTOR_DISTANCE(k.embedding, :qvec, COSINE) AS dist,
                   k.texto
            FROM kb_documentos k
            JOIN categorias c ON c.id = k.categoria_id
            WHERE k.embedding IS NOT NULL
            ORDER BY dist
            FETCH FIRST :k ROWS ONLY
            """,
            {"qvec": qvec, "k": k},
        )
        vecinos = []
        for kb_id, codigo, dist, texto_clob in cur:
            vecinos.append(
                Vecino(
                    kb_id=kb_id,
                    categoria=codigo,
                    distancia=float(dist),
                    texto=texto_clob.read() if hasattr(texto_clob, "read") else str(texto_clob),
                )
            )
        return vecinos
    finally:
        con.close()
