#!/usr/bin/env python3
"""
Full fine-tuning of roberta-base for prompt-safety classification.

Examples:
    python train.py
    python train.py --batch-size 16 --lr 2e-5
    python train.py --max-samples 200   # quick smoke-test
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import DataConfig, RobertaConfig
from data.loader import load_all, split_dataset
from evaluate import (
    compute_metrics,
    print_metrics,
    aggregate_cv_metrics,
    print_cv_metrics,
)
from model import train_roberta, predict_roberta, save_roberta

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full fine-tune roberta-base for prompt-safety classification.")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset source.")
    p.add_argument("--max-safe-samples", type=int, default=None, help="Cap safe (label=0) samples per source; overrides --max-samples for safe only.")
    p.add_argument("--cache-dir", default="./cache")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bert-model", default="roberta-base")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-seq-len", type=int, default=256)
    return p.parse_args()


def _make_cfg(args) -> RobertaConfig:
    return RobertaConfig(
        model_name=args.bert_model,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        patience=args.patience,
        cv_folds=args.cv_folds,
        output_dir=str(args.output_dir),
    )


def _cv(args, X_trainval, y_trainval) -> list[dict]:
    cfg = _make_cfg(args)
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
        logger.info("── CV fold %d/%d ──────────────", fold + 1, args.cv_folds)
        X_tr = [X_trainval[i] for i in tr_idx]
        y_tr = [y_trainval[i] for i in tr_idx]
        X_val = [X_trainval[i] for i in val_idx]
        y_val = [y_trainval[i] for i in val_idx]

        model = train_roberta(cfg, X_tr, y_tr, X_val, y_val)
        preds = predict_roberta(model, X_val, tokenizer=tokenizer, cfg=cfg, batch_size=cfg.batch_size * 2)
        m = compute_metrics(y_val, preds)
        print_metrics(m, title=f"Fold {fold + 1}")
        fold_metrics.append(m)

    return fold_metrics


def _train_final(args, X_trainval, y_trainval, X_val, y_val, output_dir: Path):
    cfg = _make_cfg(args)
    model = train_roberta(cfg, X_trainval, y_trainval, X_val, y_val)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    save_roberta(model, tokenizer, str(output_dir / "roberta_finetuned"))
    return model, tokenizer


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets …")
    by_label = {0: args.max_safe_samples} if args.max_safe_samples is not None else None
    df = load_all(
        max_per_source=args.max_samples,
        max_samples_by_label=by_label,
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

    trainval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    X_trainval = trainval_df["text"].tolist()
    y_trainval = trainval_df["label"].tolist()

    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()
    X_val = val_df["text"].tolist()
    y_val = val_df["label"].tolist()

    logger.info("Running %d-fold cross-validation …", args.cv_folds)
    fold_metrics = _cv(args, X_trainval, y_trainval)
    agg = aggregate_cv_metrics(fold_metrics)
    print_cv_metrics(agg)

    logger.info("Training final model on train+val …")
    model, tokenizer = _train_final(args, X_trainval, y_trainval, X_val, y_val, output_dir)
    test_preds = predict_roberta(model, X_test, tokenizer=tokenizer, cfg=_make_cfg(args))

    test_metrics = compute_metrics(y_test, test_preds)
    print_metrics(test_metrics, title="HELD-OUT TEST SET  [roberta_finetuned]")


if __name__ == "__main__":
    main()
