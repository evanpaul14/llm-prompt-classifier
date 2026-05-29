"""
Main entry point.

Usage:
    python train.py [--no-cache] [--skip-cv]

Stages:
  1. Load + merge all datasets
  2. Split into train+val (80%) and held-out test (20%)
  3. Generate embeddings (or load from cache)
  4. 5-fold CV on train+val — logs mean ± std F1/AUC across folds
  5. Evaluate fold-ensemble on held-out test set
"""

import argparse
import logging

import numpy as np
from sklearn.model_selection import train_test_split

from config import cfg
from data.loader import load_all, SAFE
from embeddings.embed import embed_texts, get_embedder
from training.cross_val import run_cross_validation
from evaluate import evaluate_ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Recompute embeddings even if cache exists")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV, go straight to test evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Max samples per source for quick smoke tests")
    parser.add_argument("--safe-cap", type=int, default=None, help="Cap safe (label=0) samples; loads all jailbreak/harmful")
    args = parser.parse_args()

    # 1. Data
    logger.info("Loading datasets...")
    max_by_label = {SAFE: args.safe_cap} if args.safe_cap else None
    df = load_all(
        balance=not bool(max_by_label),
        random_seed=cfg.random_seed,
        max_per_source=args.limit,
        max_samples_by_label=max_by_label,
    )
    logger.info(f"Total rows: {len(df)}")

    texts = df["text"].tolist()
    labels = df["label"].values.astype(np.int64)

    # 2. Train/test split — stratified, held-out test never touched during CV
    trainval_texts, test_texts, trainval_labels, test_labels = train_test_split(
        texts, labels,
        test_size=cfg.test_size,
        stratify=labels,
        random_state=cfg.random_seed,
    )
    logger.info(f"Train+val: {len(trainval_texts)} | Test: {len(test_texts)}")

    # 3. Embeddings
    embedder = get_embedder()
    use_cache = not args.no_cache

    logger.info("Embedding train+val texts...")
    trainval_emb = embed_texts(trainval_texts, model=embedder, use_cache=use_cache)

    logger.info("Embedding test texts...")
    test_emb = embed_texts(test_texts, model=embedder, use_cache=use_cache)

    # Reduce folds when dataset is tiny
    if args.limit and args.limit * 13 < 50:
        cfg.n_folds = 2

    # 4. Cross-validation
    if not args.skip_cv:
        logger.info("\nStarting 5-fold cross-validation...")
        run_cross_validation(trainval_emb, trainval_labels)

    # 5. Final evaluation on test set (ensemble of all fold checkpoints)
    logger.info("\nEvaluating ensemble on held-out test set...")
    results = evaluate_ensemble(test_emb, test_labels, n_folds=cfg.n_folds)
    logger.info(f"\nFinal Test F1={results['macro_f1']:.4f}  AUC={results['macro_auc']:.4f}")


if __name__ == "__main__":
    main()
