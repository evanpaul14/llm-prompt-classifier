"""5-fold stratified cross-validation for the FFNN classifier."""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from config import cfg
from dataset import EmbeddingDataset
from model import FFNNClassifier

logger = logging.getLogger(__name__)

_NUM_CLASSES = 2


def _train_fold(
    fold: int,
    tr_emb: np.ndarray,
    tr_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    device = cfg.device

    train_loader = DataLoader(
        EmbeddingDataset(tr_emb, tr_labels),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        EmbeddingDataset(val_emb, val_labels),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    model = FFNNClassifier().to(device)

    counts = np.bincount(tr_labels, minlength=_NUM_CLASSES)
    weights = torch.tensor(
        len(tr_labels) / (_NUM_CLASSES * counts), dtype=torch.float
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss += loss_fn(logits, y).item()
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_true.extend(y.cpu().tolist())

        avg_val = val_loss / len(val_loader)
        macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

        if cfg.lr_scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(avg_val)

        logger.info(
            "Fold %d  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f",
            fold,
            epoch + 1,
            cfg.max_epochs,
            train_loss / len(train_loader),
            avg_val,
            macro_f1,
        )

        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            patience_counter = 0
            ckpt = cfg.checkpoints_dir / f"fold_{fold}_best.pt"
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (best val_loss=%.4f)",
                    epoch + 1,
                    best_val_loss,
                )
                break

    return best_val_loss


def run_cross_validation(
    trainval_emb: np.ndarray, trainval_labels: np.ndarray
) -> None:
    skf = StratifiedKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_seed
    )
    fold_losses = []

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(trainval_emb, trainval_labels), start=1
    ):
        logger.info("── CV fold %d/%d ──────────────", fold, cfg.n_folds)
        best_loss = _train_fold(
            fold,
            trainval_emb[tr_idx],
            trainval_labels[tr_idx],
            trainval_emb[val_idx],
            trainval_labels[val_idx],
        )
        fold_losses.append(best_loss)
        logger.info("Fold %d complete — best val_loss=%.4f", fold, best_loss)

    logger.info(
        "CV complete — mean val_loss=%.4f ± %.4f",
        np.mean(fold_losses),
        np.std(fold_losses),
    )
