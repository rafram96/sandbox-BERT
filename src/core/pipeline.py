from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import oracledb

from .. import config
from . import db, llm, rag
from .classifier import Prediccion, get_clasificador
from .rag import Vecino


@dataclass
class Resultado:
    documento_id: int
    expediente: Optional[str]
    categoria_base: str
    score_base: float
    ruta: str
    escalo_llm: bool
    etiqueta_final: str
    revision_humana: bool
    vecinos: List[Vecino] = field(default_factory=list)
    justificacion_llm: str = ""


class Pipeline:
    def __init__(self, umbral: float = None, top_k: int = None):
        self.umbral = config.CONFIDENCE_THRESHOLD if umbral is None else umbral
        self.top_k = config.TOP_K if top_k is None else top_k
        print(f"[pipeline] Preparando clasificador (modo: {config.CLASSIFIER_MODE})...")
        self.clf = get_clasificador()


    def procesar(self, texto: str, expediente: Optional[str] = None) -> Resultado:
        pred: Prediccion = self.clf.classify(texto)
        conf = pred.margen if config.CONFIDENCE_METRIC == "margen" else pred.score

        if conf >= self.umbral:
            res = Resultado(
                documento_id=-1, expediente=expediente,
                categoria_base=pred.categoria, score_base=conf,
                ruta="rapida", escalo_llm=False,
                etiqueta_final=pred.categoria, revision_humana=False,
            )
        else:
            vecinos = rag.top_k(texto, k=self.top_k)
            decision = llm.decidir(texto, pred.categoria, pred.score, vecinos)
            res = Resultado(
                documento_id=-1, expediente=expediente,
                categoria_base=pred.categoria, score_base=conf,
                ruta="llm", escalo_llm=True,
                etiqueta_final=decision.etiqueta_final,
                revision_humana=True,
                vecinos=vecinos, justificacion_llm=decision.justificacion,
            )

        self._persistir(texto, res)
        return res


    def _persistir(self, texto: str, res: Resultado) -> None:
        con = db.connect()
        try:
            cur = con.cursor()
            doc_id = cur.var(oracledb.NUMBER)
            cur.execute(
                "INSERT INTO documentos (expediente, texto) VALUES (:e, :t) "
                "RETURNING id INTO :id",
                {"e": res.expediente, "t": texto, "id": doc_id},
            )
            res.documento_id = int(doc_id.getvalue()[0])

            top_k_json = None
            if res.vecinos:
                top_k_json = json.dumps(
                    [
                        {"kb_id": v.kb_id, "categoria": v.categoria,
                         "distancia": round(v.distancia, 4), "similitud": round(v.similitud, 4)}
                        for v in res.vecinos
                    ],
                    ensure_ascii=False,
                )

            cur.execute(
                """
                INSERT INTO clasificaciones
                    (documento_id, categoria_base, score_confianza, ruta,
                     escalo_llm, top_k_json, etiqueta_final, revision_humana)
                VALUES
                    (:doc, :cbase, :score, :ruta,
                     :escalo, :topk, :final, :rev)
                """,
                {
                    "doc": res.documento_id,
                    "cbase": res.categoria_base,
                    "score": res.score_base,
                    "ruta": res.ruta,
                    "escalo": 1 if res.escalo_llm else 0,
                    "topk": top_k_json,
                    "final": res.etiqueta_final,
                    "rev": 1 if res.revision_humana else 0,
                },
            )
            con.commit()
        finally:
            con.close()
