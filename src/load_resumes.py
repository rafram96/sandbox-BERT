"""Carga el dataset resume-data-pdf a corpus (resumes_kb.jsonl + resumes_test.jsonl).

PDFs escaneados: OCR con PyMuPDF + Tesseract. Normaliza carpetas a categorias,
muestrea N por categoria y hace split train/test.

    python -m src.load_resumes --per-cat 35
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from . import config

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


# --- Carga ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default=str(config.ROOT / "data" / "archive (1).zip"))
    ap.add_argument("--per-cat", type=int, default=35, help="PDFs a muestrear por categoria")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-pages", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=120, help="descarta OCR con menos texto")
    ap.add_argument("--min-cat", type=int, default=15, help="ignora categorias con menos PDFs")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    # agrupa rutas por categoria normalizada
    por_cat = defaultdict(list)
    for n in z.namelist():
        if n.lower().endswith(".pdf") and len(n.split("/")) > 2:
            por_cat[normalizar_categoria(n.split("/")[1])].append(n)
    cats = {c: sorted(v) for c, v in por_cat.items() if len(v) >= args.min_cat}
    print(f"Categorias usadas: {len(cats)}  (>= {args.min_cat} PDFs c/u)")

    kb_path = config.ROOT / "data" / "resumes_kb.jsonl"
    test_path = config.ROOT / "data" / "resumes_test.jsonl"
    n_kb = n_test = n_skip = 0
    with kb_path.open("w", encoding="utf-8") as fkb, test_path.open("w", encoding="utf-8") as ftest:
        for ci, (cat, rutas) in enumerate(sorted(cats.items()), 1):
            lbl = etiqueta(cat)
            recogidos = []
            for ruta in rutas:
                if len(recogidos) >= args.per_cat:
                    break
                try:
                    txt = ocr_pdf(z.read(ruta), args.dpi, args.max_pages)
                except Exception:
                    n_skip += 1
                    continue
                if len(txt) < args.min_chars:
                    n_skip += 1
                    continue
                recogidos.append((ruta, txt))
            # split estratificado
            corte = max(1, int(len(recogidos) * args.train_frac))
            for j, (ruta, txt) in enumerate(recogidos):
                if j < corte:
                    fkb.write(json.dumps({"categoria": lbl, "texto": txt}, ensure_ascii=False) + "\n")
                    n_kb += 1
                else:
                    rid = Path(ruta).stem + "_" + lbl
                    ftest.write(json.dumps(
                        {"id": rid, "texto": txt, "categoria_verdadera": lbl}, ensure_ascii=False) + "\n")
                    n_test += 1
            print(f"  [{ci:>2}/{len(cats)}] {lbl:<26} recogidos={len(recogidos):>2} "
                  f"(train {corte}, test {len(recogidos)-corte})")

    print(f"\nOK. KB={n_kb}  TEST={n_test}  descartados(OCR pobre)={n_skip}")
    print(f"  {kb_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
