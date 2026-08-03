"""Evalua el flujo sobre un test etiquetado (accuracy, reparto, lift de escalar).

Test JSONL: {"id": "...", "texto": "...", "categoria_verdadera": "IT"}

python evaluate.py --test data/test.jsonl [--umbral X]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src import config
from src.pipeline import Pipeline


def cargar(path: Path):
    filas = []
    for l in path.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l:
            filas.append(json.loads(l))
    return filas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, default="data/test.jsonl")
    ap.add_argument("--umbral", type=float, default=None)
    args = ap.parse_args()

    umbral = args.umbral if args.umbral is not None else config.CONFIDENCE_THRESHOLD
    filas = cargar(Path(args.test))
    print(f"== Evaluacion: {len(filas)} documentos de prueba | umbral={umbral} ==\n")

    pipe = Pipeline(umbral=umbral)

    n = len(filas)
    ok_global = 0
    fast_n = fast_ok = 0
    esc_n = esc_ok_final = esc_ok_base = 0
    por_cat_total = defaultdict(int)
    por_cat_ok = defaultdict(int)
    confusiones = defaultdict(int)   # (verdadera, predicha) cuando fallan

    for i, f in enumerate(filas, 1):
        verdad = f["categoria_verdadera"]
        res = pipe.procesar(f["texto"], expediente=f.get("id"))
        acierta = res.etiqueta_final == verdad

        ok_global += acierta
        por_cat_total[verdad] += 1
        por_cat_ok[verdad] += acierta
        if not acierta:
            confusiones[(verdad, res.etiqueta_final)] += 1

        if res.ruta == "rapida":
            fast_n += 1
            fast_ok += acierta
        else:
            esc_n += 1
            esc_ok_final += acierta
            esc_ok_base += (res.categoria_base == verdad)

        if i % 25 == 0:
            print(f"  ... {i}/{n}")

    def pct(a, b):
        return f"{(100.0 * a / b):.1f}%" if b else "n/a"

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"Accuracy GLOBAL              : {pct(ok_global, n)}  ({ok_global}/{n})")
    print(f"Ruta rapida                 : {fast_n} docs  | accuracy {pct(fast_ok, fast_n)}")
    print(f"Escalaron a LLM             : {esc_n} docs  | accuracy {pct(esc_ok_final, esc_n)}")
    print("-" * 60)
    print("LIFT de escalar (solo en los que escalaron):")
    print(f"  accuracy prediccion BASE  : {pct(esc_ok_base, esc_n)}")
    print(f"  accuracy etiqueta FINAL   : {pct(esc_ok_final, esc_n)}")
    delta = (esc_ok_final - esc_ok_base)
    signo = "+" if delta >= 0 else ""
    print(f"  -> RAG+LLM corrigio en neto: {signo}{delta} documentos")
    print("-" * 60)
    print("Accuracy por categoria (peores primero):")
    filas_cat = sorted(por_cat_total, key=lambda c: por_cat_ok[c] / por_cat_total[c])
    for c in filas_cat:
        print(f"  {c:<28} {pct(por_cat_ok[c], por_cat_total[c]):>7}  ({por_cat_ok[c]}/{por_cat_total[c]})")
    if confusiones:
        print("-" * 60)
        print("Top confusiones (verdadera -> predicha):")
        for (v, p), cnt in sorted(confusiones.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {v:<24} -> {p:<24} x{cnt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
