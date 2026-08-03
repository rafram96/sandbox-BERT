"""Clasificador por centroide mas cercano sobre los embeddings del corpus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from . import db, embeddings


@dataclass
class Prediccion:
    categoria: str
    score: float                 # softmax de la mejor clase (diluye con #clases)
    ranking: List[Tuple[str, float]]  # (categoria, score) ordenado desc
    margen: float = 0.0          # coseno top1 - top2 (confianza indep. de #clases)
    top1_sim: float = 0.0        # coseno crudo al mejor centroide


class ClasificadorLigero:
    def __init__(self, temperatura: float = 0.10):
        self.temperatura = temperatura
        self._codigos: List[str] = []
        self._centroides: np.ndarray | None = None  # (n_cat, dim)

    def entrenar_desde_bd(self) -> "ClasificadorLigero":
        """Carga embeddings etiquetados de Oracle y arma los centroides."""
        con = db.connect()
        try:
            cur = con.cursor()
            cur.execute(
                """
                SELECT c.codigo, k.embedding
                FROM kb_documentos k
                JOIN categorias c ON c.id = k.categoria_id
                WHERE k.embedding IS NOT NULL
                """
            )
            por_cat: Dict[str, List[np.ndarray]] = {}
            for codigo, emb in cur:
                vec = np.array(emb, dtype="float32")
                por_cat.setdefault(codigo, []).append(vec)
        finally:
            con.close()

        if not por_cat:
            raise RuntimeError(
                "No hay embeddings en kb_documentos. Corre primero: python -m src.ingest"
            )

        self._codigos = sorted(por_cat.keys())
        cents = []
        for codigo in self._codigos:
            m = np.vstack(por_cat[codigo]).mean(axis=0)
            n = np.linalg.norm(m)
            cents.append(m / n if n else m)
        self._centroides = np.vstack(cents).astype("float32")
        return self

    def classify(self, texto: str) -> Prediccion:
        if self._centroides is None:
            raise RuntimeError("Clasificador no entrenado. Llama entrenar_desde_bd().")
        q = embeddings.embed_one(texto, is_query=True)  # ya viene L2-normalizado
        sims = self._centroides @ q                     # coseno vs cada centroide
        orden_sims = np.argsort(-sims)
        top1_sim = float(sims[orden_sims[0]])
        top2_sim = float(sims[orden_sims[1]]) if len(sims) > 1 else 0.0
        margen = top1_sim - top2_sim                    # confianza indep. de #clases
        # softmax (se conserva para el ranking, aunque diluye con muchas clases)
        logits = sims / max(self.temperatura, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        orden = np.argsort(-probs)
        ranking = [(self._codigos[i], float(probs[i])) for i in orden]
        return Prediccion(
            categoria=self._codigos[orden_sims[0]], score=float(probs[orden_sims[0]]),
            ranking=ranking, margen=margen, top1_sim=top1_sim,
        )
