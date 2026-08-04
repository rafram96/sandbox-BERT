"""Pipeline ES de punta a punta: descarga el zip -> OCR + split -> traduce KB y test.
Al final NOTIFICA todos los errores (OCR descartado + docs que fallaron al traducir).

    python run_es_pipeline.py --per-cat 40 --model qwen2.5:7b

Requiere: Tesseract + Ollama (con el modelo) corriendo. El zip se descarga de
Kaggle si no esta en data/ (necesita 'pip install kaggle' + ~/.kaggle/kaggle.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from src import config, load_resumes, translate


def _es(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_es.jsonl"))


def asegurar_zip(zip_path: str | None, slug: str, skip_download: bool) -> str:
    data = config.ROOT / "data"
    data.mkdir(exist_ok=True)
    if zip_path and Path(zip_path).exists():
        print(f"[1/4] zip presente: {zip_path}")
        return zip_path
    existentes = sorted(data.glob("*.zip"))
    if existentes:
        print(f"[1/4] usando zip existente: {existentes[0]}")
        return str(existentes[0])
    if skip_download:
        raise SystemExit("[1/4] No hay zip en data/ y --skip-download esta activo.")
    print(f"[1/4] descargando {slug} de Kaggle...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(data)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise SystemExit(
            "[1/4] No pude descargar con kaggle. Opciones:\n"
            "  - pip install kaggle y coloca ~/.kaggle/kaggle.json (token de Kaggle), o\n"
            "  - descarga el zip manualmente y dejalo en data/.\n"
            f"  Detalle: {e}"
        )
    zips = sorted(data.glob("*.zip"))
    if not zips:
        raise SystemExit("[1/4] Kaggle no dejo ningun .zip en data/.")
    print(f"[1/4] descargado: {zips[0]}")
    return str(zips[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="ruta al zip (si no, se busca/descarga en data/)")
    ap.add_argument("--slug", default="hadikp/resume-data-pdf", help="dataset de Kaggle")
    ap.add_argument("--per-cat", type=int, default=35, help="PDFs por categoria a OCR-ear")
    ap.add_argument("--workers", type=int, default=1, help="procesos de OCR en paralelo")
    ap.add_argument("--model", default=config.OLLAMA_MODEL, help="modelo Ollama para traducir")
    ap.add_argument("--max-chars", type=int, default=800, help="recorte de texto al traducir")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    # 1. zip
    zip_path = asegurar_zip(args.zip, args.slug, args.skip_download)

    # 2. OCR + split
    print("\n[2/4] OCR + split...")
    ocr = load_resumes.cargar(zip_path, per_cat=args.per_cat, workers=args.workers)

    # 3. traducir KB
    print("\n[3/4] Traduciendo KB...")
    t_kb = time.time()
    kb = translate.traducir_archivo(ocr["kb_path"], _es(ocr["kb_path"]), args.model, args.max_chars)
    t_kb = time.time() - t_kb

    # 4. traducir test
    print("\n[4/4] Traduciendo test...")
    t_te = time.time()
    te = translate.traducir_archivo(ocr["test_path"], _es(ocr["test_path"]), args.model, args.max_chars)
    t_te = time.time() - t_te

    # --- Reporte final de errores ---
    errores = {
        "ocr_descartados": ocr["errores"],
        "traduccion_kb": kb["errores"],
        "traduccion_test": te["errores"],
    }
    total = sum(len(v) for v in errores.values())

    print("\n" + "=" * 60)
    print("RESUMEN DEL PIPELINE")
    print("=" * 60)
    print(f"Categorias: {ocr['n_cats']}  |  KB: {ocr['n_kb']}  |  TEST: {ocr['n_test']}")
    print(f"Salidas ES: {_es(ocr['kb_path'])}")
    print(f"            {_es(ocr['test_path'])}")
    print("-" * 60)
    print("TIEMPOS")
    print(f"  OCR:            {ocr['ocr_seg'] / 60:6.1f} min")
    print(f"  Traduccion KB:  {t_kb / 60:6.1f} min")
    print(f"  Traduccion test:{t_te / 60:6.1f} min")
    print(f"  TOTAL:          {(time.time() - t0) / 60:6.1f} min")
    print("-" * 60)
    print("ERRORES")
    print(f"  OCR descartados (PDF ilegible/vacio): {len(errores['ocr_descartados'])}")
    print(f"  Traduccion KB fallida:                {len(errores['traduccion_kb'])}")
    print(f"  Traduccion test fallida:              {len(errores['traduccion_test'])}")

    # muestra hasta 10 de cada tipo en consola
    for etapa, items in errores.items():
        for it in items[:10]:
            print(f"    [{etapa}] {it}")
        if len(items) > 10:
            print(f"    [{etapa}] ... y {len(items) - 10} mas")

    log = config.ROOT / "data" / "pipeline_errores.json"
    log.write_text(json.dumps(errores, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 60)
    print(f"Detalle completo de errores en: {log}")
    print("Sin errores." if total == 0 else f"TOTAL errores: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
