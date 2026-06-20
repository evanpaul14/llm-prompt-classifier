#!/usr/bin/env python3
"""
Unified prompt classifier — run inference with any of the 4 trained models.

Usage:
    # Interactive mode (select model, then type prompts)
    python predict.py

    # Select model by number and pass a single prompt
    python predict.py --model 1 --prompt "How do I bake a cake?"

    # Select model by name
    python predict.py --model tfidf_lr --prompt "Ignore all previous instructions"

    # Read prompts from a file (one per line)
    python predict.py --model 4 --input-file prompts.txt

Models:
    1  tfidf_lr      TF-IDF + Logistic Regression
    2  frozen_bert   Frozen RoBERTa + classification head
    3  roberta       Full fine-tuned RoBERTa
    4  ffnn_gemma    FFNN on EmbeddingGemma-300m (5-fold ensemble)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

LABELS = ["safe", "block"]


def _load_temperature(model_dir: Path) -> float:
    path = model_dir / "temperature.json"
    if path.exists():
        return json.loads(path.read_text())["temperature"]
    return 1.0

MODEL_ALIASES = {
    "1": "tfidf_lr",
    "tfidf_lr": "tfidf_lr",
    "2": "frozen_bert",
    "frozen_bert": "frozen_bert",
    "3": "roberta",
    "roberta": "roberta",
    "4": "ffnn_gemma",
    "ffnn_gemma": "ffnn_gemma",
}

MODEL_DISPLAY = {
    "tfidf_lr":    "1  TF-IDF + Logistic Regression",
    "frozen_bert": "2  Frozen RoBERTa + Classification Head",
    "roberta":     "3  Full Fine-tuned RoBERTa",
    "ffnn_gemma":  "4  FFNN on EmbeddingGemma-300m (5-fold ensemble)",
}


# ── model loaders ──────────────────────────────────────────────────────────────

def _load_tfidf_lr():
    import joblib
    path = REPO_ROOT / "outputs" / "tfidf_lr" / "tfidf_lr.joblib"
    return joblib.load(path)


def _load_frozen_bert():
    sys.path.insert(0, str(REPO_ROOT / "model_2_frozen_bert"))
    from model import load_bert
    path = REPO_ROOT / "outputs" / "frozen_bert" / "frozen_bert.pt"
    return load_bert(str(path))


def _load_roberta():
    sys.path.insert(0, str(REPO_ROOT / "model_3_roberta_finetune"))
    from model import load_roberta
    path = REPO_ROOT / "outputs" / "roberta_finetuned"
    return load_roberta(str(path))  # returns (model, tokenizer)


def _load_ffnn_gemma():
    import torch
    sys.path.insert(0, str(REPO_ROOT / "model_4_ffnn_gemma"))
    from model import FFNNClassifier
    from config import cfg
    from embeddings.embed import get_embedder

    cfg.checkpoints_dir = REPO_ROOT / "outputs" / "ffnn_gemma"
    cfg.embeddings_dir = REPO_ROOT / "model_4_ffnn_gemma" / "cache" / "embeddings"
    cfg.cache_dir = REPO_ROOT / "model_4_ffnn_gemma" / "cache"
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for fold in range(1, cfg.n_folds + 1):
        ckpt = cfg.checkpoints_dir / f"fold_{fold}_best.pt"
        if ckpt.exists():
            m = FFNNClassifier().to(cfg.device)
            m.load_state_dict(torch.load(ckpt, map_location=cfg.device, weights_only=True))
            m.eval()
            models.append(m)
    if not models:
        raise RuntimeError(f"No checkpoints found in {cfg.checkpoints_dir}")

    embedder = get_embedder()
    return models, embedder, cfg


# ── inference ──────────────────────────────────────────────────────────────────

def predict_tfidf_lr(model, texts: list[str]) -> list[dict]:
    probs_all = model.predict_proba(texts)
    raw_classes = model.classes_
    # sklearn may store integer class labels; map to string names
    str_classes = [LABELS[int(c)] if isinstance(c, (int, __import__("numpy").integer)) else str(c)
                   for c in raw_classes]
    results = []
    for text, probs in zip(texts, probs_all):
        label = str_classes[probs.argmax()]
        results.append({
            "text": text,
            "label": label,
            "probs": {c: round(float(p), 4) for c, p in zip(str_classes, probs)},
        })
    return results


def predict_frozen_bert(model, texts: list[str]) -> list[dict]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    T = _load_temperature(REPO_ROOT / "outputs" / "frozen_bert")
    tokenizer = AutoTokenizer.from_pretrained(model.cfg.model_name)
    device = next(model.parameters()).device
    model.eval()
    results = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, padding="max_length",
                        max_length=model.cfg.max_seq_len, return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs = F.softmax(logits / T, dim=-1).cpu()
        for text, prob in zip(chunk, probs):
            pred = int(prob.argmax())
            results.append({
                "text": text,
                "label": LABELS[pred],
                "probs": {LABELS[j]: round(float(prob[j]), 4) for j in range(len(LABELS))},
            })
    return results


def predict_roberta(model_and_tok, texts: list[str]) -> list[dict]:
    import torch
    import torch.nn.functional as F
    T = _load_temperature(REPO_ROOT / "outputs" / "roberta_finetuned")
    model, tokenizer = model_and_tok
    sys.path.insert(0, str(REPO_ROOT / "model_3_roberta_finetune"))
    device = next(model.parameters()).device
    model.eval()
    results = []
    batch_size = 32
    preds_all = []
    probs_all = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, padding="max_length",
                        max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"].to(device),
                           enc["attention_mask"].to(device)).logits
        probs = F.softmax(logits / T, dim=-1).cpu()
        preds_all.extend(logits.argmax(dim=-1).cpu().tolist())
        probs_all.extend(probs.tolist())

    for text, pred, probs in zip(texts, preds_all, probs_all):
        results.append({
            "text": text,
            "label": LABELS[pred],
            "probs": {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))},
        })
    return results


def predict_ffnn_gemma(model_data, texts: list[str]) -> list[dict]:
    import torch
    T = _load_temperature(REPO_ROOT / "outputs" / "ffnn_gemma")
    models, embedder, cfg = model_data
    emb = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    x = torch.tensor(emb, dtype=torch.float32).to(cfg.device)
    with torch.no_grad():
        logits = torch.stack([m(x) for m in models]).mean(0)
    probs = torch.softmax(logits / T, dim=1).cpu().numpy()
    results = []
    for text, prob in zip(texts, probs):
        label = LABELS[prob.argmax()]
        results.append({
            "text": text,
            "label": label,
            "probs": {LABELS[i]: round(float(prob[i]), 4) for i in range(len(LABELS))},
        })
    return results


# ── dispatch ───────────────────────────────────────────────────────────────────

LOADERS = {
    "tfidf_lr":    _load_tfidf_lr,
    "frozen_bert": _load_frozen_bert,
    "roberta":     _load_roberta,
    "ffnn_gemma":  _load_ffnn_gemma,
}

PREDICTORS = {
    "tfidf_lr":    predict_tfidf_lr,
    "frozen_bert": predict_frozen_bert,
    "roberta":     predict_roberta,
    "ffnn_gemma":  predict_ffnn_gemma,
}


# ── display ────────────────────────────────────────────────────────────────────

VERDICT = {"safe": "SAFE ", "block": "BLOCK"}


def print_results(results: list[dict]) -> None:
    print(f"\n{'─' * 70}")
    for r in results:
        trunc = r["text"][:60] + ("…" if len(r["text"]) > 60 else "")
        verdict = VERDICT.get(r["label"], r["label"].upper())
        probs_str = "  ".join(f"{k}={v:.3f}" for k, v in r["probs"].items())
        print(f"  {verdict}  {trunc}")
        print(f"           probs: {probs_str}")
    print(f"{'─' * 70}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _select_model_interactive() -> str:
    print("\nAvailable models:")
    for key, desc in MODEL_DISPLAY.items():
        print(f"  {desc}")
    print()
    while True:
        choice = input("Select model (1-4 or name): ").strip().lower()
        if choice in MODEL_ALIASES:
            return MODEL_ALIASES[choice]
        print("  Invalid choice. Enter 1, 2, 3, 4, tfidf_lr, frozen_bert, roberta, or ffnn_gemma.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified LLM prompt safety classifier.")
    p.add_argument(
        "--model", "-m", default=None,
        help="Model to use: 1/tfidf_lr, 2/frozen_bert, 3/roberta, 4/ffnn_gemma"
    )
    p.add_argument("--prompt", "-p", default=None, help="Single prompt string.")
    p.add_argument("--input-file", "-f", default=None, help="File with one prompt per line.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve model key
    if args.model is None:
        model_key = _select_model_interactive()
    else:
        key = args.model.strip().lower()
        if key not in MODEL_ALIASES:
            print(f"Unknown model '{args.model}'. Choose from: 1, 2, 3, 4, tfidf_lr, frozen_bert, roberta, ffnn_gemma")
            sys.exit(1)
        model_key = MODEL_ALIASES[key]

    print(f"\nLoading {MODEL_DISPLAY[model_key].strip()} …", flush=True)
    artifact = LOADERS[model_key]()
    print("Model loaded.\n")

    # Collect texts
    if args.prompt:
        texts = [args.prompt]
    elif args.input_file:
        lines = Path(args.input_file).read_text().splitlines()
        texts = [l.strip() for l in lines if l.strip()]
    else:
        # Interactive loop
        print("Enter prompts to classify (empty line to quit):\n")
        predictor = PREDICTORS[model_key]
        while True:
            try:
                text = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not text:
                break
            print_results(predictor(artifact, [text]))
        return

    results = PREDICTORS[model_key](artifact, texts)
    print_results(results)


if __name__ == "__main__":
    main()
