"""Clasificadores: por centroide (embeddings) o ModernBERT/XLM-R fine-tuneado.

get_clasificador() elige segun config.CLASSIFIER_MODE ('centroide' | 'finetuned').
Ambos exponen la misma interfaz: entrenar_desde_bd() y classify(texto) -> Prediccion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .. import config
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
                "No hay embeddings en kb_documentos. Corre primero: python -m src.corpus.ingest"
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


class ClasificadorFineTuned:
    """Carga un modelo fine-tuneado (finetune.py) y clasifica con su cabeza softmax.

    La confianza (score) es la probabilidad softmax de la clase ganadora -> ya es
    una senal calibrada para la puerta (con CONFIDENCE_METRIC=softmax).
    """

    def __init__(self, model_path: str, max_len: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.max_len = max_len
        print(f"[classifier] Cargando modelo fine-tuneado de {model_path} ...")
        # Los checkpoints del Trainer no siempre guardan el tokenizer. Ojo: cargarlo
        # de un dir sin vocab NO lanza excepcion, devuelve uno vacio (vocab_size~5)
        # que tokeniza todo como <unk> -> hay que validar el vocabulario.
        self.tok = None
        try:
            t = AutoTokenizer.from_pretrained(model_path)
            if t.vocab_size >= 1000:
                self.tok = t
        except Exception:
            pass
        if self.tok is None:
            print(f"[classifier] Tokenizer ausente/vacio en {model_path}; "
                  f"uso {config.FT_TOKENIZER}")
            self.tok = AutoTokenizer.from_pretrained(config.FT_TOKENIZER)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # el config puede traer las claves como str ('0') o int (0) -> normaliza a int
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        print(f"[classifier] Listo en {self.device}  ({len(self.id2label)} clases).")

    def entrenar_desde_bd(self) -> "ClasificadorFineTuned":
        return self  # ya viene entrenado

    def classify(self, texto: str) -> Prediccion:
        torch = self._torch
        enc = self.tok(str(texto), truncation=True, max_length=self.max_len,
                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        orden = np.argsort(-probs)
        ranking = [(str(self.id2label[int(i)]), float(probs[i])) for i in orden]
        top1 = float(probs[orden[0]])
        top2 = float(probs[orden[1]]) if len(orden) > 1 else 0.0
        return Prediccion(
            categoria=ranking[0][0], score=top1, ranking=ranking,
            margen=top1 - top2, top1_sim=top1,
        )


def get_clasificador():
    """Factory: devuelve el clasificador segun config.CLASSIFIER_MODE."""
    if config.CLASSIFIER_MODE == "finetuned":
        return ClasificadorFineTuned(config.FT_MODEL_PATH)
    return ClasificadorLigero().entrenar_desde_bd()
