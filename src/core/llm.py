from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from .. import config
from .rag import Vecino


@dataclass
class DecisionLLM:
    etiqueta_final: str
    justificacion: str
    puntajes: dict


def _mock(categoria_base: str, score_base: float, vecinos: List[Vecino]) -> DecisionLLM:
    puntajes = defaultdict(float)
    for v in vecinos:
        puntajes[v.categoria] += max(v.similitud, 0.0)
    if categoria_base:
        puntajes[categoria_base] += float(score_base)
    ganadora = max(puntajes.items(), key=lambda kv: kv[1])[0]
    return DecisionLLM(
        etiqueta_final=ganadora,
        justificacion=f"[mock] voto ponderado -> {ganadora}",
        puntajes=dict(puntajes),
    )


def _ollama_generate(prompt: str, timeout: float = 60.0) -> str:
    payload = json.dumps({
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def _construir_prompt(texto: str, categoria_base: str, score_base: float,
                      vecinos: List[Vecino], candidatas: List[str]) -> str:
    vecinos_txt = "\n".join(
        f"  - {v.categoria} (similitud {v.similitud:.2f}): {v.texto[:180]}"
        for v in vecinos
    )
    lista = ", ".join(candidatas)
    return (
        "Eres un asistente de clasificacion de documentos (CVs / hojas de vida).\n"
        "Elige la MEJOR categoria profesional para el CV de abajo.\n\n"
        f"TEXTO DEL CV (puede tener ruido de OCR):\n{texto[:1500]}\n\n"
        f"Un clasificador ligero predijo: {categoria_base} (confianza {score_base:.2f}).\n"
        f"Los CVs etiquetados mas similares (RAG Top-K) son:\n{vecinos_txt}\n\n"
        f"Categorias validas para elegir: {lista}\n\n"
        "Responde SOLO con un codigo de categoria de esa lista, nada mas."
    )


def _match_label(respuesta: str, candidatas: List[str]) -> str | None:
    r = respuesta.strip().upper()
    for c in candidatas:
        if c.upper() == r:
            return c
    for c in candidatas:
        if re.search(rf"\b{re.escape(c.upper())}\b", r):
            return c
    return None


def decidir(texto: str, categoria_base: str, score_base: float,
            vecinos: List[Vecino]) -> DecisionLLM:
    if config.USE_MOCK_LLM:
        return _mock(categoria_base, score_base, vecinos)


    candidatas = []
    for c in [categoria_base] + [v.categoria for v in vecinos]:
        if c and c not in candidatas:
            candidatas.append(c)

    try:
        prompt = _construir_prompt(texto, categoria_base, score_base, vecinos, candidatas)
        respuesta = _ollama_generate(prompt)
        elegido = _match_label(respuesta, candidatas)
        if elegido is None:
            return DecisionLLM(
                etiqueta_final=categoria_base,
                justificacion=f"[llm sin etiqueta valida: {respuesta.strip()[:60]!r}] -> base",
                puntajes={},
            )
        coincide = "confirma" if elegido == categoria_base else "corrige"
        return DecisionLLM(
            etiqueta_final=elegido,
            justificacion=f"[llm {config.OLLAMA_MODEL}] {coincide} -> {elegido}",
            puntajes={},
        )
    except Exception as e:
        m = _mock(categoria_base, score_base, vecinos)
        m.justificacion = f"[llm error: {type(e).__name__}] fallback {m.justificacion}"
        return m
