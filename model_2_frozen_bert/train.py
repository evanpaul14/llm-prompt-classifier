#!/usr/bin/env python3
"""
Training entry point — frozen RoBERTa + classification head only.

Examples:
    python train.py
    python train.py --batch-size 32 --lr 1e-3
    python train.py --max-samples 200   # quick smoke-test
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import DataConfig, BertConfig
from data.loader import load_all, split_dataset
from evaluate import (
    compute_metrics,
    print_metrics,
    aggregate_cv_metrics,
    print_cv_metrics,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train frozen-RoBERTa prompt-safety classifier.")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset source.")
    p.add_argument("--max-safe", type=int, default=None, help="Cap total safe (label=0) samples after loading.")
    p.add_argument("--cache-dir", default="./cache")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bert-model", default="roberta-base")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--head-hidden-size", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.1)
    return p.parse_args()


# ── BERT helpers ──────────────────────────────────────────────────────────────

def _make_bert_cfg(args) -> BertConfig:
    return BertConfig(
        model_name=args.bert_model,
        frozen=True,
        head_hidden_size=args.head_hidden_size,
        head_dropout=args.head_dropout,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        patience=args.patience,
        cv_folds=args.cv_folds,
        output_dir=str(args.output_dir),
    )


def _cv_bert(args, X_trainval, y_trainval) -> list[dict]:
    from model import train_bert, predict_bert
    from transformers import AutoTokenizer

    cfg = _make_bert_cfg(args)
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
        logger.info("── CV fold %d/%d ──────────────", fold + 1, args.cv_folds)
        X_tr = [X_trainval[i] for i in tr_idx]
        y_tr = [y_trainval[i] for i in tr_idx]
        X_val = [X_trainval[i] for i in val_idx]
        y_val = [y_trainval[i] for i in val_idx]

        model = train_bert(cfg, X_tr, y_tr, X_val, y_val)
        preds = predict_bert(model, X_val, tokenizer=tokenizer, batch_size=cfg.batch_size * 2)
        m = compute_metrics(y_val, preds)
        print_metrics(m, title=f"Fold {fold + 1}")
        fold_metrics.append(m)

    return fold_metrics


def _train_final_bert(args, X_trainval, y_trainval, X_val, y_val, output_dir: Path):
    from model import train_bert, save_bert

    cfg = _make_bert_cfg(args)
    model = train_bert(cfg, X_trainval, y_trainval, X_val, y_val)
    save_bert(model, str(output_dir / "frozen_bert.pt"))
    return model


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────
    logger.info("Loading datasets …")
    max_by_label = {0: args.max_safe} if args.max_safe is not None else None
    df = load_all(
        max_per_source=args.max_samples,
        max_samples_by_label=max_by_label,
    )
    data_cfg = DataConfig(random_seed=args.seed)
    train_df, val_df, test_df = split_dataset(
        df,
        test_size=data_cfg.test_size,
        val_size=data_cfg.val_size,
        random_seed=data_cfg.random_seed,
    )

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d  (total: %d)",
        len(train_df), len(val_df), len(test_df), len(df),
    )
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["label"].value_counts().sort_index().to_dict()
        logger.info("  %s class distribution: %s", split_name, dist)

    # Combined train+val for CV (test is held out)
    trainval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    X_trainval = trainval_df["text"].tolist()
    y_trainval = trainval_df["label"].tolist()

    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()
    # val used as the early-stop set when training the final model
    X_val = val_df["text"].tolist()
    y_val = val_df["label"].tolist()

    from model import predict_bert
    from transformers import AutoTokenizer

    # ── 5-fold CV ────────────────────────────────────────────────────────
    logger.info("Running %d-fold cross-validation …", args.cv_folds)
    fold_metrics = _cv_bert(args, X_trainval, y_trainval)
    agg = aggregate_cv_metrics(fold_metrics)
    print_cv_metrics(agg)

    # ── train final model on train+val, evaluate on test ─────────────────
    logger.info("Training final model on train+val …")
    model = _train_final_bert(args, X_trainval, y_trainval, X_val, y_val, output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    test_preds = predict_bert(model, X_test, tokenizer=tokenizer)

    test_metrics = compute_metrics(y_test, test_preds)
    print_metrics(test_metrics, title="HELD-OUT TEST SET  [frozen_bert]")


if __name__ == "__main__":
    main()
