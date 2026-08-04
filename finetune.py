"""Fine-tunea un encoder (ModernBERT / XLM-R / e5) como CLASIFICADOR de CVs.

Reemplaza el clasificador por centroides por una cabeza de clasificacion entrenada
de verdad (cross-entropy, backprop). Compara contra el baseline de centroides (~55%).

Requiere GPU para ser practico. En el server:
    pip install torch transformers numpy          # torch con soporte CUDA

    # Ingles (mas datos, ModernBERT):
    python finetune.py --train data/servidor/traducciones/resumes_kb.jsonl \
                       --test  data/servidor/traducciones/resumes_test.jsonl \
                       --model answerdotai/ModernBERT-base --epochs 4

    # Espanol (base multilingue):
    python finetune.py --train data/resumes_kb_es_sub.jsonl \
                       --test  data/resumes_test_es_sub.jsonl \
                       --model xlm-roberta-base --epochs 5

Salida: accuracy en test + modelo guardado en --out (reutilizable en inferencia).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)


def _load(path: str, text_key: str, label_key: str):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [(r[text_key], r[label_key]) for r in rows]


class CVDataset(Dataset):
    def __init__(self, pairs, tok, l2id, max_len):
        self.pairs, self.tok, self.l2id, self.max_len = pairs, tok, l2id, max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        texto, label = self.pairs[i]
        enc = self.tok(texto, truncation=True, max_length=self.max_len)
        enc["labels"] = self.l2id[label]
        return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="jsonl con {texto, categoria}")
    ap.add_argument("--test", required=True, help="jsonl con {texto, categoria_verdadera}")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--patience", type=int, default=2,
                    help="early stopping: para si no mejora en N epochs (0 = desactiva)")
    ap.add_argument("--out", default="ft-model")
    args = ap.parse_args()

    train = _load(args.train, "texto", "categoria")
    test = _load(args.test, "texto", "categoria_verdadera")

    labels = sorted({l for _, l in train})
    l2id = {l: i for i, l in enumerate(labels)}
    id2l = {i: l for l, i in l2id.items()}
    # descarta test con categorias no vistas en train
    test = [(t, l) for t, l in test if l in l2id]
    print(f"train={len(train)}  test={len(test)}  clases={len(labels)}  "
          f"GPU={'si' if torch.cuda.is_available() else 'NO (sera lento)'}")

    tok = AutoTokenizer.from_pretrained(args.model)
    # ModernBERT usa torch.compile/Triton internamente (reference_compile); en Windows
    # Triton no anda y el backward se vuelve lentisimo -> lo desactivamos.
    extra = {"reference_compile": False} if "modernbert" in args.model.lower() else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2l, label2id=l2id, **extra)

    ds_train = CVDataset(train, tok, l2id, args.max_len)
    ds_test = CVDataset(test, tok, l2id, args.max_len)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": float((preds == p.label_ids).mean())}

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),   # mixed precision en GPU (Turing+): mas rapido
        logging_steps=50,
        report_to="none",
    )
    callbacks = []
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds_train, eval_dataset=ds_test,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    t0 = time.time()
    trainer.train()
    train_seg = time.time() - t0

    te0 = time.time()
    res = trainer.evaluate()
    eval_seg = time.time() - te0

    print("\n" + "=" * 56)
    print(f"ACCURACY FINE-TUNED: {res['eval_accuracy']*100:.1f}%   "
          f"(baseline centroides ~55%)")
    print(f"TIEMPO entrenamiento: {train_seg:.1f} s  ({train_seg/60:.1f} min)")
    print(f"TIEMPO evaluacion:    {eval_seg:.1f} s")
    print(f"Dispositivo: {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 56)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Modelo guardado en {args.out}/")


if __name__ == "__main__":
    main()
