"""Clasifica documentos largos por trozos con un modelo de ventana corta.

Parte el documento en chunks, clasifica cada uno y combina las probabilidades.

    py -m src.training.chunking --test data/resumes_test_es_sub.jsonl \
                       --model-path ft-model-xlmr --tokenizer xlm-roberta-base
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_CHUNK = 512


def _load(path: str, limit: int):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        t = r.get("texto")
        if not isinstance(t, str):
            t = " ".join(map(str, t)) if isinstance(t, list) else ("" if t is None else str(t))
        t = t.strip()
        lbl = r.get("categoria_verdadera")
        if t and lbl:
            out.append((t, str(lbl)))
        if limit and len(out) >= limit:
            break
    return out


def _cargar_tokenizer(model_path: str, base: str):
    # si el dir no trae vocab, from_pretrained devuelve un tokenizer vacio
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        if tok.vocab_size >= 1000:
            return tok
    except Exception:
        pass
    return AutoTokenizer.from_pretrained(base)


def _ventanas(ids, chunk_size, stride, cls_id, sep_id, modo):
    extra = (cls_id is not None) + (sep_id is not None)
    cuerpo = chunk_size - extra
    paso = max(cuerpo - stride, 1)
    trozos = [ids[i:i + cuerpo] for i in range(0, max(len(ids), 1), paso)]
    trozos = [t for t in trozos if t] or [[]]
    if modo == "extremos" and len(trozos) > 3:
        trozos = [trozos[0], trozos[len(trozos) // 2], trozos[-1]]
    salida = []
    for t in trozos:
        seq = ([cls_id] if cls_id is not None else []) + list(t) + \
              ([sep_id] if sep_id is not None else [])
        salida.append(seq)
    return salida


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="jsonl con {texto, categoria_verdadera}")
    ap.add_argument("--model-path", required=True, help="dir del modelo fine-tuneado")
    ap.add_argument("--tokenizer", required=True, help="tokenizer base de respaldo")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=0, help="solape entre chunks (0 = sin solape)")
    ap.add_argument("--agregacion", choices=["max", "mean"], default="mean")
    ap.add_argument("--paginas", choices=["todas", "extremos"], default="todas",
                    help="extremos = primer chunk + del medio + ultimo")
    ap.add_argument("--limit", type=int, default=0, help="0 = todos los documentos")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    chunk_size = min(args.chunk_size, MAX_CHUNK)
    docs = _load(args.test, args.limit)
    if not docs:
        raise SystemExit(f"Sin documentos validos en {args.test}")

    tok = _cargar_tokenizer(args.model_path, args.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # claves de id2label llegan como str desde el json
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    print(f"docs={len(docs)}  clases={len(id2label)}  chunk={chunk_size}  "
          f"stride={args.stride}  agregacion={args.agregacion}  paginas={args.paginas}  "
          f"dispositivo={device}")

    t0 = time.time()
    chunks, dueno = [], []
    for i, (texto, _) in enumerate(docs):
        ids = tok(texto, add_special_tokens=False, truncation=False)["input_ids"]
        for seq in _ventanas(ids, chunk_size, args.stride, cls_id, sep_id, args.paginas):
            chunks.append(seq)
            dueno.append(i)

    probs = np.zeros((len(chunks), len(id2label)), dtype="float32")
    for ini in range(0, len(chunks), args.batch):
        lote = chunks[ini:ini + args.batch]
        ancho = max(len(s) for s in lote)
        input_ids = torch.tensor([s + [pad_id] * (ancho - len(s)) for s in lote])
        mask = torch.tensor([[1] * len(s) + [0] * (ancho - len(s)) for s in lote])
        with torch.no_grad():
            logits = model(input_ids=input_ids.to(device),
                           attention_mask=mask.to(device)).logits
        probs[ini:ini + len(lote)] = torch.softmax(logits, dim=-1).cpu().numpy()

    aciertos = aciertos_primero = 0
    for i, (_, verdadera) in enumerate(docs):
        filas = probs[np.array(dueno) == i]
        if args.agregacion == "mean":
            combinado = filas.mean(axis=0)
        else:
            combinado = filas[int(np.argmax(filas.max(axis=1)))]
        if id2label[int(np.argmax(combinado))] == verdadera:
            aciertos += 1
        if id2label[int(np.argmax(filas[0]))] == verdadera:
            aciertos_primero += 1

    seg = time.time() - t0
    acc = aciertos / len(docs) * 100
    acc_primero = aciertos_primero / len(docs) * 100

    print("\n" + "=" * 56)
    print(f"ACCURACY por chunks ({args.agregacion}): {acc:.1f}%")
    print(f"ACCURACY solo primer chunk:      {acc_primero:.1f}%")
    print(f"Diferencia: {acc - acc_primero:+.1f} puntos")
    print(f"Chunks procesados: {len(chunks)}  ({len(chunks)/len(docs):.1f} por documento)")
    print(f"TIEMPO total: {seg:.1f} s  ({seg/len(docs):.2f} s por documento)")
    print("=" * 56)


if __name__ == "__main__":
    main()
