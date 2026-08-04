"""Traduce el campo 'texto' de un corpus JSONL EN->ES con un LLM local (Ollama).

Requisitos donde lo corras: Ollama levantado con el modelo, y este repo.

    python -m src.translate --in data/resumes_kb.jsonl   --out data/resumes_kb_es.jsonl
    python -m src.translate --in data/resumes_test.jsonl --out data/resumes_test_es.jsonl

Opciones:
    --model M        modelo Ollama (default: OLLAMA_MODEL del .env)
    --max-chars N    recorta el texto de entrada (default 4000)
    --limit N        traduce solo los primeros N (0 = todos)
    --resume         continua un --out a medias (salta lo ya traducido)
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from . import config

_PREAMBULO = re.compile(
    r"^\s*(aqu[ií] est[aá][^\n:]*:|here is[^\n:]*:|traducci[oó]n[^\n:]*:|translation[^\n:]*:)\s*",
    re.IGNORECASE,
)


def _generate(prompt: str, model: str, timeout: float = 180.0) -> str:
    payload = json.dumps({
        "model": model,
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


def traducir(texto: str, model: str, max_chars: int) -> str:
    prompt = (
        "Eres un traductor profesional. Traduce el TEXTO al espanol neutro.\n"
        "Responde unicamente con la traduccion, sin introduccion ni comentarios.\n\n"
        f"TEXTO:\n{texto[:max_chars]}\n\nTRADUCCION:"
    )
    out = _generate(prompt, model).strip()
    return _PREAMBULO.sub("", out).strip()


def _cargar(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def traducir_archivo(inp: str, out: str, model: str, max_chars: int = 4000,
                     limit: int = 0, resume: bool = False) -> dict:
    """Traduce un JSONL. Devuelve stats y la lista de docs que fallaron."""
    filas = _cargar(Path(inp))
    if limit:
        filas = filas[:limit]

    hechos = 0
    out_path = Path(out)
    if resume and out_path.exists():
        hechos = len(_cargar(out_path))
        print(f"Resume: ya hay {hechos} traducidos, continuo desde ahi.")

    modo = "a" if (resume and hechos) else "w"
    total = len(filas)
    print(f"Traduciendo {total - hechos} docs con {model} -> {out}")

    n_ok = 0
    errores = []  # {doc, id, error}
    t0 = time.time()
    with open(out_path, modo, encoding="utf-8") as f:
        for i in range(hechos, total):
            fila = filas[i]
            try:
                fila["texto"] = traducir(fila["texto"], model, max_chars)
                n_ok += 1
            except Exception as e:  # deja el texto original si Ollama falla
                errores.append({"doc": i + 1, "id": fila.get("id", ""), "error": type(e).__name__})
                print(f"  err doc {i + 1}: {type(e).__name__}")
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()
            hecho = i - hechos + 1
            if hecho % 25 == 0:
                seg = (time.time() - t0) / hecho
                falta = (total - i - 1) * seg
                print(f"  ... {i + 1}/{total}  ({seg:.1f}s/doc, ETA {falta/60:.0f} min)")

    print(f"OK. traducidos={n_ok} errores={len(errores)} -> {out}")
    return {"n_ok": n_ok, "errores": errores, "out": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model", default=config.OLLAMA_MODEL)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    traducir_archivo(args.inp, args.out, args.model, args.max_chars, args.limit, args.resume)


if __name__ == "__main__":
    main()
