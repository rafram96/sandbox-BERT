from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from .. import config

_backend = None


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class ModernBERTBackend:
    name = "ModernBERT"

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        print(f"[embeddings] Cargando {model_name} (ModernBERT)... (1a vez descarga de HuggingFace)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[embeddings] ModernBERT listo en {self.device}.")

    def embed(self, textos: List[str], batch_size: int = 16) -> np.ndarray:
        torch = self._torch
        out = []
        for i in range(0, len(textos), batch_size):
            lote = textos[i : i + batch_size]
            enc = self.tokenizer(
                lote, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                res = self.model(**enc)

            hidden = res.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = (summed / counts).cpu().numpy().astype("float32")
            out.append(emb)
        return _l2_normalize(np.vstack(out))


class HashBackend:
    name = "hash-fallback"

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, textos: List[str], batch_size: int = 16) -> np.ndarray:
        vecs = np.zeros((len(textos), self.dim), dtype="float32")
        for i, t in enumerate(textos):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        return _l2_normalize(vecs)


def _get_backend():
    global _backend
    if _backend is not None:
        return _backend
    if config.FORCE_HASH_EMBEDDING:
        print("[embeddings] FORCE_HASH_EMBEDDING=1 -> usando embedding hash (no ModernBERT).")
        _backend = HashBackend(config.EMBEDDING_DIM)
        return _backend
    try:
        _backend = ModernBERTBackend(config.EMBEDDING_MODEL)
    except Exception as e:
        print(f"[embeddings] AVISO: no se pudo cargar ModernBERT ({e}).")
        print("[embeddings] Cae al embedding hash. Instala torch+transformers para usar ModernBERT.")
        _backend = HashBackend(config.EMBEDDING_DIM)
    return _backend


def backend_name() -> str:
    return _get_backend().name


def _aplicar_prefijo(textos: List[str], is_query: bool) -> List[str]:

    pref = config.EMBED_QUERY_PREFIX if is_query else config.EMBED_DOC_PREFIX
    if not pref:
        return textos
    return [f"{pref} {t}" for t in textos]


def embed(textos: List[str], is_query: bool = False) -> np.ndarray:


    if isinstance(textos, str):
        textos = [textos]
    return _get_backend().embed(_aplicar_prefijo(textos, is_query))


def embed_one(texto: str, is_query: bool = False) -> np.ndarray:
    return embed([texto], is_query=is_query)[0]
