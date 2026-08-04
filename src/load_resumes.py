"""Carga el dataset resume-data-pdf a corpus (resumes_kb.jsonl + resumes_test.jsonl).

PDFs escaneados: OCR con PyMuPDF + Tesseract. Normaliza carpetas a categorias,
muestrea N por categoria y hace split train/test.

    python -m src.load_resumes --per-cat 35
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from . import config

# 1 hilo de OpenMP por proceso: evita que Tesseract se sobre-suscriba cuando
# lo paralelizamos a nivel de proceso (los workers heredan este env en spawn).
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

# --- Normalizacion de categorias -------------------------------------------
_ALIAS = {
    "hr": "human resources", "it": "information technology", "datascience": "data science",
    "nse": "network security engineer", "dot": "dotnet developer", "dot net developer": "dotnet developer",
    "webdesigning": "web designing", "designing": "web designing", "design": "designer",
    "agricultural": "agriculture", "managment": "management", "operationmanager": "operations manager",
    "operation manager": "operations manager", "consult": "consultant", "public": "public relations",
    "food": "food beverages", "digital": "digital media", "healthfitness": "health fitness",
    "architects": "architect", "civilengineer": "civil engineer", "mechanicalengineer": "mechanical engineer",
    "electrical engineering": "electrical engineer", "electricalengineer": "electrical engineer",
    "pythondeveloper": "python developer", "javadeveloper": "java developer",
    "devopsengineer": "devops engineer", "dev ops engineer": "devops engineer",
    "sapdeveloper": "sap developer", "sql": "sql developer", "etl": "etl developer",
    "react": "react developer", "businessanalyst": "business analyst", "pbo": "pmo",
}


def normalizar_categoria(folder: str) -> str:
    s = folder.replace("_", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).lower()   # parte camelCase
    s = re.sub(r"\bresumes?\b", " ", s)                    # quita 'resume'/'resumes'
    s = re.sub(r"\s+", " ", s).strip()
    return _ALIAS.get(s, s)


def etiqueta(cat: str) -> str:
    """Codigo compacto en MAYUSCULAS para guardar como categoria."""
    return re.sub(r"[^A-Z0-9]+", "_", cat.upper()).strip("_")


# --- OCR --------------------------------------------------------------------
def ocr_pdf(data: bytes, dpi: int, max_pages: int) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        partes = []
        for i in range(min(max_pages, doc.page_count)):
            pix = doc[i].get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            partes.append(pytesseract.image_to_string(img, lang="eng"))
    finally:
        doc.close()
    return re.sub(r"\s+", " ", " ".join(partes)).strip()


# --- OCR paralelo (pool de procesos) ----------------------------------------
_ZIP = None  # zip abierto una vez por worker (no se comparte entre procesos)


def _init_worker(zip_path: str) -> None:
    global _ZIP
    os.environ["OMP_THREAD_LIMIT"] = "1"
    _ZIP = zipfile.ZipFile(zip_path)


def _ocr_task(args):
    """Corre en un worker. args = (member, lbl, dpi, max_pages)."""
    member, lbl, dpi, max_pages = args
    try:
        txt = ocr_pdf(_ZIP.read(member), dpi, max_pages)
        return (member, lbl, txt, None)
    except Exception as e:  # noqa: BLE001
        return (member, lbl, None, type(e).__name__)


# --- Carga ------------------------------------------------------------------
def cargar(zip_path: str, per_cat: int = 35, train_frac: float = 0.7, dpi: int = 200,
           max_pages: int = 2, min_chars: int = 120, min_cat: int = 15,
           workers: int = 1) -> dict:
    """OCR (paralelo si workers>1) + normalizacion + split.

    Devuelve stats y la lista de PDFs descartados (fallo de OCR o texto corto).
    """
    z = zipfile.ZipFile(zip_path)
    por_cat = defaultdict(list)
    for n in z.namelist():
        if n.lower().endswith(".pdf") and len(n.split("/")) > 2:
            por_cat[normalizar_categoria(n.split("/")[1])].append(n)
    z.close()
    cats = {c: sorted(v) for c, v in por_cat.items() if len(v) >= min_cat}

    # plan de tareas: los primeros per_cat PDFs de cada categoria
    tareas = [(ruta, etiqueta(cat), dpi, max_pages)
              for cat, rutas in sorted(cats.items()) for ruta in rutas[:per_cat]]
    print(f"Categorias: {len(cats)}  |  PDFs a OCR: {len(tareas)}  |  workers: {workers}")

    # OCR
    total = len(tareas)
    t0 = time.time()

    def _progreso(k: int) -> None:
        el = time.time() - t0
        rate = k / el if el else 0
        eta = (total - k) / rate if rate else 0
        print(f"  ... {k}/{total}  |  transcurrido {el/60:.1f} min  |  "
              f"{rate:.1f} pdf/s  |  ETA {eta/60:.1f} min")

    resultados = []
    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers,
                                    initializer=_init_worker, initargs=(zip_path,)) as ex:
            for k, r in enumerate(ex.map(_ocr_task, tareas, chunksize=8), 1):
                resultados.append(r)
                if k % 200 == 0:
                    _progreso(k)
    else:
        _init_worker(zip_path)
        for k, t in enumerate(tareas, 1):
            resultados.append(_ocr_task(t))
            if k % 100 == 0:
                _progreso(k)

    ocr_seg = time.time() - t0

    # agrupa por categoria, filtra fallos/texto corto
    por_lbl = defaultdict(list)
    errores = []  # {ruta, motivo}
    for member, lbl, txt, err in resultados:
        if err is not None:
            errores.append({"ruta": member, "motivo": err})
        elif len(txt) < min_chars:
            errores.append({"ruta": member, "motivo": "texto_corto"})
        else:
            por_lbl[lbl].append((member, txt))

    # split estratificado train/test (determinista: ordenado por member)
    kb_path = config.ROOT / "data" / "resumes_kb.jsonl"
    test_path = config.ROOT / "data" / "resumes_test.jsonl"
    n_kb = n_test = 0
    with kb_path.open("w", encoding="utf-8") as fkb, test_path.open("w", encoding="utf-8") as ftest:
        for lbl in sorted(por_lbl):
            recogidos = sorted(por_lbl[lbl])
            corte = max(1, int(len(recogidos) * train_frac))
            for j, (member, txt) in enumerate(recogidos):
                if j < corte:
                    fkb.write(json.dumps({"categoria": lbl, "texto": txt}, ensure_ascii=False) + "\n")
                    n_kb += 1
                else:
                    rid = Path(member).stem + "_" + lbl
                    ftest.write(json.dumps(
                        {"id": rid, "texto": txt, "categoria_verdadera": lbl}, ensure_ascii=False) + "\n")
                    n_test += 1

    print(f"\nOK. KB={n_kb}  TEST={n_test}  descartados(OCR pobre)={len(errores)}  "
          f"|  OCR en {ocr_seg/60:.1f} min")
    return {"n_kb": n_kb, "n_test": n_test, "n_cats": len(cats), "ocr_seg": ocr_seg,
            "kb_path": str(kb_path), "test_path": str(test_path), "errores": errores}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default=str(config.ROOT / "data" / "archive (1).zip"))
    ap.add_argument("--per-cat", type=int, default=35, help="PDFs a muestrear por categoria")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-pages", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=120, help="descarta OCR con menos texto")
    ap.add_argument("--min-cat", type=int, default=15, help="ignora categorias con menos PDFs")
    ap.add_argument("--workers", type=int, default=1, help="procesos de OCR en paralelo")
    args = ap.parse_args()
    cargar(args.zip, args.per_cat, args.train_frac, args.dpi, args.max_pages,
           args.min_chars, args.min_cat, args.workers)


if __name__ == "__main__":
    main()
