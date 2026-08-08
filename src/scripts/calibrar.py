from __future__ import annotations

import argparse
import json
import statistics as st

from ..core.classifier import get_clasificador


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--target", type=float, default=0.95, help="accuracy minima en ruta rapida")
    args = ap.parse_args()

    clf = get_clasificador()
    filas = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    rows = []
    for d in filas:
        p = clf.classify(d["texto"])
        rows.append((p.margen, p.categoria == d["categoria_verdadera"]))

    margs = [m for m, _ in rows]
    total = len(rows)
    base_acc = sum(ok for _, ok in rows) / total
    print(f"\nTest: {total} docs  |  accuracy base (clasificador solo): {base_acc*100:.1f}%")
    print(f"MARGEN: min={min(margs):.3f}  mediana={st.median(margs):.3f}  max={max(margs):.3f}\n")
    print(f"{'umbral':>7} {'ruta_rapida':>12} {'acc_rapida':>11} {'escalan':>8} {'acc_base_esc':>13}")

    candidatos = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    sugerido = None
    for thr in candidatos:
        fast = [ok for m, ok in rows if m >= thr]
        esc = [ok for m, ok in rows if m < thr]
        fa = sum(fast) / len(fast) if fast else 0.0
        ea = sum(esc) / len(esc) if esc else 0.0
        print(f"{thr:>7.2f} {len(fast):>5}/{total:<5} {fa*100:>9.0f}% {len(esc):>8} {ea*100:>11.0f}%")
        if sugerido is None and fast and fa >= args.target:
            sugerido = thr

    print()
    if sugerido is not None:
        print(f"SUGERIDO: CONFIDENCE_THRESHOLD={sugerido}  "
              f"(ruta rapida con accuracy >= {args.target:.0%}, maxima cobertura)")
    else:
        print(f"Ningun umbral alcanza {args.target:.0%} en ruta rapida. "
              f"Baja --target o revisa el corpus/embeddings.")


if __name__ == "__main__":
    main()
