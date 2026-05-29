"""
Final evaluation on the held-out test set.

Loads the best checkpoint from each fold, ensembles their logits (average),
then reports full classification metrics for all three classes.
"""

import logging

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from config import cfg
from data.dataset import EmbeddingDataset
from data.loader import CLASS_NAMES
from models.ffnn import FFNNClassifier, NUM_CLASSES

logger = logging.getLogger(__name__)


def evaluate_ensemble(
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    n_folds: int = cfg.n_folds,
) -> dict:
    dataset = EmbeddingDataset(test_embeddings, test_labels)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    device  = cfg.device

    all_fold_logits = []
    for fold in range(1, n_folds + 1):
        ckpt = cfg.checkpoints_dir / f"fold_{fold}_best.pt"
        if not ckpt.exists():
            logger.warning(f"Checkpoint {ckpt} not found — skipping fold {fold}")
            continue
        model = FFNNClassifier().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        fold_logits = []
        with torch.no_grad():
            for X, _ in loader:
                fold_logits.append(model(X.to(device)).cpu())
        all_fold_logits.append(torch.cat(fold_logits).numpy())

    avg_logits = np.mean(all_fold_logits, axis=0)                  # (N, 3)
    probs      = torch.softmax(torch.tensor(avg_logits), dim=1).numpy()
    preds      = avg_logits.argmax(axis=1)
    labels     = test_labels

    macro_f1  = f1_score(labels, preds, average="macro", zero_division=0)
    macro_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    cm        = confusion_matrix(labels, preds)
    report    = classification_report(labels, preds, target_names=CLASS_NAMES)

    logger.info(f"\nTest-set results ({len(labels):,} samples):")
    logger.info(f"  macro-F1={macro_f1:.4f}  macro-AUC={macro_auc:.4f}")
    logger.info(f"  Confusion matrix:\n{cm}")
    logger.info(f"\n{report}")

    return {
        "macro_f1": macro_f1, "macro_auc": macro_auc,
        "confusion_matrix": cm, "probs": probs, "preds": preds,
    }
