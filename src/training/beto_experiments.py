"""Evaluacion reproducible de BETO directo y BETO con chunking.

Genera todos los artefactos bajo ``output/beto`` por defecto:

* separacion estratificada train/validacion (el test nunca selecciona modelos),
* checkpoint BETO directo (primeros ``max_len`` tokens),
* checkpoint BETO entrenado con todos los chunks,
* evaluacion directa en test completo/corto/largo,
* chunking: primer chunk, Mean/Max y todos/extremos,
* metricas JSON, resumen CSV, predicciones y matrices de confusion.

Uso en el servidor:

    py -m src.training.beto_experiments

Para comprobar datos y tamanos sin entrenar:

    py -m src.training.beto_experiments --prepare-only
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

@dataclass
class Documento:
    texto: str
    etiqueta: str
    doc_id: str


class SequenceDataset(Dataset):
    """Dataset de secuencias ya tokenizadas, con padding fijo."""

    def __init__(self, sequences: Sequence[Sequence[int]], labels: Sequence[int],
                 pad_id: int, max_len: int):
        if len(sequences) != len(labels):
            raise ValueError("Cantidad distinta de secuencias y etiquetas")
        self.sequences = sequences
        self.labels = labels
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict:
        seq = list(self.sequences[index])[: self.max_len]
        padding = self.max_len - len(seq)
        return {
            "input_ids": seq + [self.pad_id] * padding,
            "attention_mask": [1] * len(seq) + [0] * padding,
            "labels": self.labels[index],
        }


def _read_jsonl(path: Path, label_key: str) -> list[Documento]:
    docs: list[Documento] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido en {path}, linea {line_no}: {exc}") from exc
            text = row.get("texto", "")
            if not isinstance(text, str):
                text = " ".join(map(str, text)) if isinstance(text, list) else str(text or "")
            label = row.get(label_key)
            if not text.strip() or label is None:
                raise ValueError(f"Texto/etiqueta vacio en {path}, linea {line_no}")
            docs.append(Documento(
                texto=text.strip(),
                etiqueta=str(label),
                doc_id=str(row.get("id", line_no)),
            ))
    if not docs:
        raise ValueError(f"No hay documentos validos en {path}")
    return docs


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stratified_split(docs: Sequence[Documento], val_ratio: float,
                      seed: int) -> tuple[list[int], list[int]]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, doc in enumerate(docs):
        by_label[doc.etiqueta].append(index)
    rng = random.Random(seed)
    train: list[int] = []
    val: list[int] = []
    for label in sorted(by_label):
        indexes = by_label[label][:]
        rng.shuffle(indexes)
        n_val = max(1, int(round(len(indexes) * val_ratio)))
        if n_val >= len(indexes):
            n_val = max(len(indexes) - 1, 0)
        val.extend(indexes[:n_val])
        train.extend(indexes[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _tokenize_docs(docs: Sequence[Documento], tokenizer, desc: str) -> list[list[int]]:
    result = []
    for doc in tqdm(docs, desc=desc, unit="doc"):
        result.append(tokenizer.encode(
            doc.texto, add_special_tokens=False, truncation=False, verbose=False,
        ))
    return result


def _special_ids(tokenizer) -> tuple[int | None, int | None, int]:
    cls_id = tokenizer.cls_token_id
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return cls_id, sep_id, pad_id


def _add_special(ids: Sequence[int], cls_id: int | None,
                 sep_id: int | None) -> list[int]:
    return (([cls_id] if cls_id is not None else []) + list(ids) +
            ([sep_id] if sep_id is not None else []))


def _windows(ids: Sequence[int], max_len: int, stride: int,
             cls_id: int | None, sep_id: int | None) -> list[list[int]]:
    special_count = int(cls_id is not None) + int(sep_id is not None)
    body = max_len - special_count
    if body < 1:
        raise ValueError("max_len demasiado pequeno para los tokens especiales")
    if not 0 <= stride < body:
        raise ValueError(f"stride debe estar entre 0 y {body - 1}")
    step = body - stride
    raw = [list(ids[start : start + body]) for start in range(0, max(len(ids), 1), step)]
    raw = [chunk for chunk in raw if chunk] or [[]]
    return [_add_special(chunk, cls_id, sep_id) for chunk in raw]


def _direct_sequences(tokenized: Sequence[Sequence[int]], max_len: int,
                      cls_id: int | None, sep_id: int | None) -> list[list[int]]:
    special_count = int(cls_id is not None) + int(sep_id is not None)
    body = max_len - special_count
    return [_add_special(ids[:body], cls_id, sep_id) for ids in tokenized]


def _expand_chunks(indexes: Sequence[int], tokenized: Sequence[Sequence[int]],
                   docs: Sequence[Documento], label2id: dict[str, int], max_len: int,
                   stride: int, cls_id: int | None, sep_id: int | None
                   ) -> tuple[list[list[int]], list[int], list[int]]:
    sequences: list[list[int]] = []
    labels: list[int] = []
    owners: list[int] = []
    for doc_index in indexes:
        chunks = _windows(tokenized[doc_index], max_len, stride, cls_id, sep_id)
        sequences.extend(chunks)
        labels.extend([label2id[docs[doc_index].etiqueta]] * len(chunks))
        owners.extend([doc_index] * len(chunks))
    return sequences, labels, owners


def _confusion(y_true: Sequence[int], y_pred: Sequence[int], n_labels: int) -> np.ndarray:
    matrix = np.zeros((n_labels, n_labels), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def _metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]) -> dict:
    matrix = _confusion(y_true, y_pred, len(labels))
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    per_category = []
    f1_values = []
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        support = int(matrix[i, :].sum())
        predicted = int(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_category.append({
            "categoria": label,
            "support": support,
            "correctos": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return {
        "n_documentos": total,
        "correctos": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "por_categoria": per_category,
    }


def _trainer_metrics(labels: Sequence[str]):
    def compute(eval_prediction) -> dict:
        predictions = np.argmax(eval_prediction.predictions, axis=1)
        result = _metrics(eval_prediction.label_ids, predictions, labels)
        return {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}
    return compute


def _save_confusion(path: Path, y_true: Sequence[int], y_pred: Sequence[int],
                    labels: Sequence[str]) -> None:
    matrix = _confusion(y_true, y_pred, len(labels))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["verdadera/predicha", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *map(int, row)])


def _new_model(model_name: str, labels: Sequence[str]):
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in id2label.items()}
    model_config = AutoConfig.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id,
    )
    if hasattr(model_config, "reference_compile"):
        model_config.reference_compile = False
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)


def _train_or_load(name: str, model_name: str, tokenizer, labels: Sequence[str],
                   train_ds: Dataset, val_ds: Dataset, output_root: Path,
                   epochs: float, batch: int, eval_batch: int, grad_accum: int,
                   lr: float, patience: int, seed: int):
    model_dir = output_root / "models" / name
    checkpoint_dir = output_root / "checkpoints" / name
    if (model_dir / "config.json").exists():
        print(f"[{name}] Modelo final existente; se reutiliza: {model_dir}")
        return AutoModelForSequenceClassification.from_pretrained(model_dir), 0.0, True

    model = _new_model(model_name, labels)
    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
    )
    callbacks = []
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_trainer_metrics(labels),
        callbacks=callbacks,
    )
    checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    if checkpoint:
        print(f"[{name}] Reanudando checkpoint: {checkpoint}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.time() - started
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    _write_json(output_root / "training" / f"{name}_history.json", trainer.state.log_history)
    train_info = {
        "nombre": name,
        "segundos": elapsed,
        "minutos": elapsed / 60,
        "epoca_final": trainer.state.epoch,
        "mejor_checkpoint": trainer.state.best_model_checkpoint,
        "mejor_metrica": trainer.state.best_metric,
        "gpu_peak_mb": (
            torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
        ),
    }
    _write_json(output_root / "training" / f"{name}.json", train_info)
    return trainer.model, elapsed, False


def _predict_sequences(model, sequences: Sequence[Sequence[int]], pad_id: int,
                       batch_size: int, device: str, desc: str) -> np.ndarray:
    model.to(device)
    model.eval()
    n_labels = int(model.config.num_labels)
    output = np.zeros((len(sequences), n_labels), dtype=np.float32)
    for start in tqdm(range(0, len(sequences), batch_size), desc=desc, unit="batch"):
        batch = sequences[start : start + batch_size]
        width = max(len(seq) for seq in batch)
        ids = torch.tensor(
            [list(seq) + [pad_id] * (width - len(seq)) for seq in batch], dtype=torch.long,
            device=device,
        )
        mask = torch.tensor(
            [[1] * len(seq) + [0] * (width - len(seq)) for seq in batch], dtype=torch.long,
            device=device,
        )
        context = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
        with torch.inference_mode(), context:
            logits = model(input_ids=ids, attention_mask=mask).logits
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        output[start : start + len(batch)] = probabilities
    return output


def _scope_indexes(chunk_counts: Sequence[int], scope: str) -> list[int]:
    if scope == "short":
        return [i for i, count in enumerate(chunk_counts) if count == 1]
    if scope == "long":
        return [i for i, count in enumerate(chunk_counts) if count > 1]
    return list(range(len(chunk_counts)))


def _save_evaluation(output_root: Path, variant: str, docs: Sequence[Documento],
                     labels: Sequence[str], label2id: dict[str, int], predictions: np.ndarray,
                     confidences: np.ndarray, chunk_counts: Sequence[int], seconds: float,
                     summary_rows: list[dict]) -> None:
    variant_dir = output_root / "evaluations" / variant
    y_true_all = np.array([label2id[doc.etiqueta] for doc in docs], dtype=np.int64)
    scopes = {}
    for scope in ("all", "short", "long"):
        indexes = _scope_indexes(chunk_counts, scope)
        y_true = y_true_all[indexes]
        y_pred = predictions[indexes]
        result = _metrics(y_true, y_pred, labels)
        result["segundos_inferencia_total"] = seconds
        result["segundos_por_documento"] = seconds / len(docs) if docs else 0
        scopes[scope] = result
        _save_confusion(
            variant_dir / f"confusion_{scope}.csv", y_true, y_pred, labels,
        )
        summary_rows.append({
            "modelo": "BETO",
            "variante": variant,
            "alcance": scope,
            "documentos": result["n_documentos"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "segundos_inferencia": seconds,
        })
    _write_json(variant_dir / "metrics.json", scopes)
    _write_jsonl(variant_dir / "predictions.jsonl", (
        {
            "id": doc.doc_id,
            "categoria_verdadera": doc.etiqueta,
            "categoria_predicha": labels[int(predictions[i])],
            "correcto": labels[int(predictions[i])] == doc.etiqueta,
            "confianza": float(confidences[i]),
            "chunks": int(chunk_counts[i]),
        }
        for i, doc in enumerate(docs)
    ))


def _aggregate_chunk_probs(probabilities: np.ndarray, doc_slices: Sequence[slice],
                           mode: str, extremes: bool) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    confidences = []
    for doc_slice in doc_slices:
        rows = probabilities[doc_slice]
        if extremes and len(rows) > 3:
            rows = rows[[0, len(rows) // 2, len(rows) - 1]]
        combined = rows.mean(axis=0) if mode == "mean" else rows.max(axis=0)
        pred = int(np.argmax(combined))
        predictions.append(pred)
        confidences.append(float(combined[pred]))
    return np.array(predictions, dtype=np.int64), np.array(confidences, dtype=np.float32)


def _write_summary(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["modelo", "variante", "alcance", "documentos", "accuracy",
              "macro_f1", "segundos_inferencia"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentos BETO directo y chunking")
    parser.add_argument("--train", default="data/espanol/resumes_kb_es_full.jsonl")
    parser.add_argument("--test", default="data/espanol/resumes_test_es_full.jsonl")
    parser.add_argument("--output", default="output/beto")
    parser.add_argument("--model", default="dccuchile/bert-base-spanish-wwm-cased")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-chunked", action="store_true")
    args = parser.parse_args()

    _seed_everything(args.seed)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train)
    test_path = Path(args.test)
    train_docs = _read_jsonl(train_path, "categoria")
    test_docs = _read_jsonl(test_path, "categoria_verdadera")
    labels = sorted({doc.etiqueta for doc in train_docs})
    if set(doc.etiqueta for doc in test_docs) - set(labels):
        raise ValueError("El test contiene categorias ausentes en entrenamiento")
    label2id = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cls_id, sep_id, pad_id = _special_ids(tokenizer)
    train_tokens = _tokenize_docs(train_docs, tokenizer, "Tokenizando entrenamiento")
    test_tokens = _tokenize_docs(test_docs, tokenizer, "Tokenizando test")
    train_indexes, val_indexes = _stratified_split(
        train_docs, args.validation_ratio, args.seed,
    )

    train_windows = [
        _windows(ids, args.max_len, args.stride, cls_id, sep_id) for ids in train_tokens
    ]
    test_windows = [
        _windows(ids, args.max_len, args.stride, cls_id, sep_id) for ids in test_tokens
    ]
    manifest = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "model": args.model,
        "seed": args.seed,
        "max_len": args.max_len,
        "stride": args.stride,
        "validation_ratio": args.validation_ratio,
        "documentos": {
            "train_total": len(train_docs),
            "train_ajuste": len(train_indexes),
            "validacion": len(val_indexes),
            "test": len(test_docs),
            "test_cortos": sum(len(x) == 1 for x in test_windows),
            "test_largos": sum(len(x) > 1 for x in test_windows),
        },
        "chunks": {
            "train_total": sum(len(x) for x in train_windows),
            "train_ajuste": sum(len(train_windows[i]) for i in train_indexes),
            "validacion": sum(len(train_windows[i]) for i in val_indexes),
            "test": sum(len(x) for x in test_windows),
            "promedio_train": sum(len(x) for x in train_windows) / len(train_windows),
            "promedio_test": sum(len(x) for x in test_windows) / len(test_windows),
            "maximo_train": max(map(len, train_windows)),
            "maximo_test": max(map(len, test_windows)),
        },
        "categorias": labels,
        "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    _write_json(output_root / "dataset_manifest.json", manifest)
    _write_json(output_root / "run_config.json", vars(args))
    print(json.dumps(manifest["documentos"], ensure_ascii=False, indent=2))
    print(json.dumps(manifest["chunks"], ensure_ascii=False, indent=2))
    if args.prepare_only:
        print(f"Preparacion validada. Manifiesto: {output_root / 'dataset_manifest.json'}")
        return
    if not torch.cuda.is_available():
        raise SystemExit("No hay CUDA. Usa --prepare-only o ejecuta el entrenamiento en GPU.")

    train_labels = [label2id[doc.etiqueta] for doc in train_docs]
    direct_sequences = _direct_sequences(train_tokens, args.max_len, cls_id, sep_id)
    summary_rows: list[dict] = []
    device = "cuda"

    if not args.skip_direct:
        direct_train_ds = SequenceDataset(
            [direct_sequences[i] for i in train_indexes],
            [train_labels[i] for i in train_indexes], pad_id, args.max_len,
        )
        direct_val_ds = SequenceDataset(
            [direct_sequences[i] for i in val_indexes],
            [train_labels[i] for i in val_indexes], pad_id, args.max_len,
        )
        direct_model, _, _ = _train_or_load(
            "direct", args.model, tokenizer, labels, direct_train_ds, direct_val_ds,
            output_root, args.epochs, args.batch, args.eval_batch,
            args.gradient_accumulation, args.lr, args.patience, args.seed,
        )
        test_direct = _direct_sequences(test_tokens, args.max_len, cls_id, sep_id)
        started = time.time()
        direct_probs = _predict_sequences(
            direct_model, test_direct, pad_id, args.eval_batch, device, "BETO directo",
        )
        elapsed = time.time() - started
        direct_pred = np.argmax(direct_probs, axis=1)
        direct_conf = direct_probs[np.arange(len(direct_pred)), direct_pred]
        chunk_counts = [len(x) for x in test_windows]
        _save_evaluation(
            output_root, "direct", test_docs, labels, label2id, direct_pred,
            direct_conf, chunk_counts, elapsed, summary_rows,
        )
        del direct_model
        gc.collect()
        torch.cuda.empty_cache()

    if not args.skip_chunked:
        chunk_train_seq, chunk_train_labels, _ = _expand_chunks(
            train_indexes, train_tokens, train_docs, label2id, args.max_len,
            args.stride, cls_id, sep_id,
        )
        chunk_val_seq, chunk_val_labels, _ = _expand_chunks(
            val_indexes, train_tokens, train_docs, label2id, args.max_len,
            args.stride, cls_id, sep_id,
        )
        chunk_train_ds = SequenceDataset(
            chunk_train_seq, chunk_train_labels, pad_id, args.max_len,
        )
        chunk_val_ds = SequenceDataset(
            chunk_val_seq, chunk_val_labels, pad_id, args.max_len,
        )
        chunk_model, _, _ = _train_or_load(
            "chunked", args.model, tokenizer, labels, chunk_train_ds, chunk_val_ds,
            output_root, args.epochs, args.batch, args.eval_batch,
            args.gradient_accumulation, args.lr, args.patience, args.seed,
        )
        flat_test: list[list[int]] = []
        doc_slices: list[slice] = []
        for chunks in test_windows:
            start = len(flat_test)
            flat_test.extend(chunks)
            doc_slices.append(slice(start, len(flat_test)))
        started = time.time()
        chunk_probs = _predict_sequences(
            chunk_model, flat_test, pad_id, args.eval_batch, device, "BETO chunks",
        )
        elapsed = time.time() - started
        chunk_counts = [len(x) for x in test_windows]
        variants = [
            ("chunk_first", "first", False),
            ("chunk_all_mean", "mean", False),
            ("chunk_all_max", "max", False),
            ("chunk_extremes_mean", "mean", True),
            ("chunk_extremes_max", "max", True),
        ]
        for variant, mode, extremes in variants:
            if mode == "first":
                selected = np.vstack([chunk_probs[s.start] for s in doc_slices])
                pred = np.argmax(selected, axis=1)
                conf = selected[np.arange(len(pred)), pred]
            else:
                pred, conf = _aggregate_chunk_probs(chunk_probs, doc_slices, mode, extremes)
            _save_evaluation(
                output_root, variant, test_docs, labels, label2id, pred, conf,
                chunk_counts, elapsed, summary_rows,
            )
        del chunk_model
        gc.collect()
        torch.cuda.empty_cache()

    _write_summary(output_root / "summary.csv", summary_rows)
    _write_json(output_root / "summary.json", summary_rows)
    print(f"\nResultados BETO listos en: {output_root.resolve()}")


if __name__ == "__main__":
    main()
