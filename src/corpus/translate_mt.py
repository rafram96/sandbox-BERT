"""Traduce corpus JSONL completos de ingles a espanol con MarianMT.

No requiere Ollama ni otro servidor. El modelo se descarga una vez desde
Hugging Face y luego puede ejecutarse localmente, incluso en CPU.

Ejemplos:

    py -m src.corpus.translate_mt \
        --in data/ingles/resumes_kb_en_full.jsonl \
        --out data/espanol/resumes_kb_es_full.jsonl
    py -m src.corpus.translate_mt \
        --in data/ingles/resumes_test_en_full.jsonl \
        --out data/espanol/resumes_test_es_full.jsonl --resume

Solo se traduce el campo ``texto``. Los demas campos del JSON se conservan.
Los documentos se dividen por tokens (no caracteres) y se reconstruyen sin
descartar ninguna parte del texto.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

MODEL = "Helsinki-NLP/opus-mt-en-es"
DEFAULT_CHUNK_TOKENS = 400
MAX_MODEL_TOKENS = 512


def _cargar(path: Path) -> list[dict]:
    """Carga un JSONL y muestra la linea exacta si encuentra JSON invalido."""
    filas = []
    with path.open(encoding="utf-8") as f:
        for numero, linea in enumerate(f, 1):
            if not linea.strip():
                continue
            try:
                fila = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido en {path}, linea {numero}: {exc}") from exc
            if not isinstance(fila, dict):
                raise ValueError(f"Se esperaba un objeto JSON en {path}, linea {numero}")
            filas.append(fila)
    return filas


def _unidades_texto(texto: str) -> Iterable[str]:
    """Produce parrafos/oraciones para intentar cortar en limites naturales."""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n").strip()
    for parrafo in re.split(r"\n\s*\n+", texto):
        parrafo = " ".join(parrafo.split())
        if not parrafo:
            continue
        partes = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", parrafo)
        yield from (parte.strip() for parte in partes if parte.strip())


def segmentar_por_tokens(texto: str, tokenizer, max_tokens: int) -> list[str]:
    """Divide todo el texto sin truncarlo y favorece finales de oracion.

    Una oracion excepcionalmente larga se corta por ids del tokenizer. Al
    decodificar esos ids puede cambiar espacio en blanco, pero no se elimina
    contenido util.
    """
    if not 1 <= max_tokens <= MAX_MODEL_TOKENS - 2:
        raise ValueError(
            f"--chunk-tokens debe estar entre 1 y {MAX_MODEL_TOKENS - 2}"
        )

    segmentos_ids: list[list[int]] = []
    actual: list[int] = []

    for unidad in _unidades_texto(str(texto)):
        ids = tokenizer.encode(unidad, add_special_tokens=False, verbose=False)
        if not ids:
            continue

        if len(ids) > max_tokens:
            if actual:
                segmentos_ids.append(actual)
                actual = []
            for inicio in range(0, len(ids), max_tokens):
                segmentos_ids.append(ids[inicio : inicio + max_tokens])
        elif len(actual) + len(ids) <= max_tokens:
            actual.extend(ids)
        else:
            segmentos_ids.append(actual)
            actual = ids

    if actual:
        segmentos_ids.append(actual)

    # Evita tokenizer.decode(), que falla de forma intermitente en transformers
    # 4.48.x. No se pasan los ids directamente a spm_source: Marian usa un
    # vocabulario combinado cuyos ids no coinciden con los ids internos de SPM.
    if hasattr(tokenizer, "convert_tokens_to_string"):
        return [
            tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens([int(token_id) for token_id in ids])
            ).strip()
            for ids in segmentos_ids
            if ids
        ]
    return [
        tokenizer.decode([int(token_id) for token_id in ids], skip_special_tokens=True).strip()
        for ids in segmentos_ids
        if ids
    ]


def _campos_no_texto(fila: dict) -> dict:
    return {k: v for k, v in fila.items() if k != "texto"}


def _validar_resume(entrada: list[dict], salida: list[dict], out_path: Path) -> None:
    """Evita continuar sobre una salida de otra fuente o con orden diferente."""
    if len(salida) > len(entrada):
        raise ValueError(f"{out_path} tiene mas filas que el archivo de entrada")
    for indice, (original, traducida) in enumerate(zip(entrada, salida), 1):
        if _campos_no_texto(original) != _campos_no_texto(traducida):
            raise ValueError(
                f"{out_path} no corresponde a la entrada (diferencia en fila {indice})"
            )
        if not str(traducida.get("texto", "")).strip():
            raise ValueError(f"Traduccion vacia en {out_path}, fila {indice}")


class TraductorMarian:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        chunk_tokens: int,
        beams: int,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se pidio CUDA, pero PyTorch no detecta una GPU compatible")

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.chunk_tokens = chunk_tokens
        self.beams = beams

        print(f"Cargando {model_name} en {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def traducir(self, texto: str) -> tuple[str, int]:
        segmentos = segmentar_por_tokens(texto, self.tokenizer, self.chunk_tokens)
        if not segmentos:
            return "", 0

        traducciones: list[str] = []
        for inicio in range(0, len(segmentos), self.batch_size):
            lote = segmentos[inicio : inicio + self.batch_size]
            enc = self.tokenizer(
                lote,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.device)
            if enc["input_ids"].shape[1] > MAX_MODEL_TOKENS:
                raise RuntimeError("Error interno: un fragmento excedio 512 tokens")
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **enc,
                    max_length=MAX_MODEL_TOKENS,
                    num_beams=self.beams,
                    early_stopping=self.beams > 1,
                )
            traducciones.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )

        if len(traducciones) != len(segmentos) or any(not t.strip() for t in traducciones):
            raise RuntimeError("MarianMT devolvio uno o mas fragmentos vacios")
        return "\n\n".join(t.strip() for t in traducciones), len(segmentos)


def traducir_archivo(
    inp: str,
    out: str,
    model_name: str = MODEL,
    batch_size: int = 8,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    beams: int = 1,
    device: str = "auto",
    limit: int = 0,
    resume: bool = False,
) -> dict:
    inp_path = Path(inp)
    out_path = Path(out)
    if inp_path.resolve() == out_path.resolve():
        raise ValueError("La entrada y la salida no pueden ser el mismo archivo")
    if batch_size < 1:
        raise ValueError("--batch debe ser mayor que cero")
    if beams < 1:
        raise ValueError("--beams debe ser mayor que cero")

    filas = _cargar(inp_path)
    if limit:
        filas = filas[:limit]

    salida_existente: list[dict] = []
    if resume and out_path.exists():
        salida_existente = _cargar(out_path)
        _validar_resume(filas, salida_existente, out_path)
        print(f"Resume: {len(salida_existente)} documentos ya traducidos.")
    elif out_path.exists():
        print(f"AVISO: se reemplazara la salida existente: {out_path}")

    hechos = len(salida_existente)
    pendientes = filas[hechos:]
    if not pendientes:
        print(f"Nada pendiente: {hechos}/{len(filas)} documentos ya estan traducidos.")
        return {"n_ok": 0, "n_total": len(filas), "segmentos": 0, "out": str(out_path)}

    traductor = TraductorMarian(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        chunk_tokens=chunk_tokens,
        beams=beams,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    modo = "a" if hechos else "w"
    inicio = time.time()
    segmentos_total = 0
    print(
        f"Traduciendo {len(pendientes)}/{len(filas)} documentos completos "
        f"(chunk={chunk_tokens}, batch={batch_size}, beams={beams}) -> {out_path}"
    )

    from tqdm.auto import tqdm

    barra = tqdm(
        pendientes,
        total=len(filas),
        initial=hechos,
        unit="doc",
        desc=out_path.name,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    with out_path.open(modo, encoding="utf-8", newline="\n") as f:
        for procesados, original in enumerate(barra, 1):
            fila = dict(original)
            try:
                traduccion, n_segmentos = traductor.traducir(str(fila.get("texto", "")))
                if str(fila.get("texto", "")).strip() and not traduccion:
                    raise RuntimeError("La traduccion completa quedo vacia")
                fila["texto"] = traduccion
            except Exception as exc:
                posicion = hechos + procesados
                raise RuntimeError(
                    f"Fallo el documento {posicion}; corrige el problema y usa --resume"
                ) from exc

            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()
            segmentos_total += n_segmentos
            barra.set_postfix(fragmentos=segmentos_total, refresh=False)

    minutos = (time.time() - inicio) / 60
    print(f"OK. Traducidos={len(pendientes)} en {minutos:.1f} min -> {out_path}")
    return {
        "n_ok": len(pendientes),
        "n_total": len(filas),
        "segmentos": segmentos_total,
        "minutos": minutos,
        "out": str(out_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Traduce JSONL completos EN->ES localmente con MarianMT"
    )
    ap.add_argument("--in", dest="inp", required=True, help="JSONL original en ingles")
    ap.add_argument("--out", dest="out", required=True, help="nuevo JSONL en espanol")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=8, help="fragmentos por lote")
    ap.add_argument(
        "--chunk-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help="tokens de entrada por fragmento (maximo 510)",
    )
    ap.add_argument(
        "--beams", type=int, default=1,
        help="1 es rapido en CPU; 4 prioriza calidad y tarda mas",
    )
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--limit", type=int, default=0, help="0 procesa todos")
    ap.add_argument("--resume", action="store_true", help="continua una salida parcial valida")
    args = ap.parse_args()

    traducir_archivo(
        inp=args.inp,
        out=args.out,
        model_name=args.model,
        batch_size=args.batch,
        chunk_tokens=args.chunk_tokens,
        beams=args.beams,
        device=args.device,
        limit=args.limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
