"""
BERT-based classifiers (Models 3–4):
  - frozen=True  → all RoBERTa encoder layers frozen; only classification head trains
  - frozen=False → full end-to-end fine-tuning

Classification head:
  If head_hidden_size > 0 : [CLS] (768) → Linear(768, hidden) → ReLU → Dropout → Linear(hidden, num_labels)
  If head_hidden_size == 0: [CLS] (768) → Linear(768, num_labels)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

from config import BertConfig

logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def _make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Stratified sampling: each batch mirrors overall class distribution."""
    counts = np.bincount(labels)
    weights_per_class = 1.0 / counts
    sample_weights = torch.tensor([weights_per_class[lbl] for lbl in labels])
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


# ── Classification Head ───────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, head_hidden: int, dropout: float, num_labels: int = 2):
        super().__init__()
        if head_hidden > 0:
            self.layers = nn.Sequential(
                nn.Linear(hidden_size, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_labels),
            )
        else:
            self.layers = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ── Main model ────────────────────────────────────────────────────────────────

class BertPromptClassifier(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_name)

        if cfg.frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen — only classification head will train.")

        hidden_size = self.encoder.config.hidden_size
        self.head = ClassificationHead(
            hidden_size, cfg.head_hidden_size, cfg.head_dropout, num_labels=cfg.num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.head(cls_repr)


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state:
            model.load_state_dict(self.best_state)


# ── Trainer ───────────────────────────────────────────────────────────────────

def _get_device(cfg: BertConfig) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_bert(
    cfg: BertConfig,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
) -> BertPromptClassifier:
    device = _get_device(cfg)
    logger.info("Training on %s (frozen=%s)", device, cfg.frozen)

    train_counts = np.bincount(train_labels, minlength=cfg.num_labels)
    val_counts = np.bincount(val_labels, minlength=cfg.num_labels)
    logger.info(
        "Dataset sizes — train: %d %s  val: %d %s",
        len(train_labels), train_counts.tolist(),
        len(val_labels), val_counts.tolist(),
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = PromptDataset(train_texts, train_labels, tokenizer, cfg.max_seq_len)
    val_ds = PromptDataset(val_texts, val_labels, tokenizer, cfg.max_seq_len)

    sampler = _make_weighted_sampler(train_labels)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2)

    model = BertPromptClassifier(cfg).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Parameters — trainable: %s / %s (%.1f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    logger.info("Scheduler — total steps: %d  warmup steps: %d", total_steps, warmup_steps)

    # class-weighted loss to handle imbalance
    class_weights = torch.tensor(len(train_labels) / (cfg.num_labels * train_counts), dtype=torch.float).to(device)
    logger.info("Class weights: %s", [f"{w:.3f}" for w in class_weights.tolist()])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    stopper = EarlyStopping(patience=cfg.patience)

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, attention_mask)
                val_loss += loss_fn(logits, labels).item()
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_acc = train_correct / len(train_labels)
        val_acc = val_correct / len(val_labels)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %d/%d  lr=%.2e  train_loss=%.4f  train_acc=%.3f  val_loss=%.4f  val_acc=%.3f  patience=%d/%d",
            epoch + 1, cfg.num_epochs, current_lr, avg_train, train_acc, avg_val, val_acc,
            stopper.counter, cfg.patience,
        )

        if stopper.step(avg_val, model):
            logger.info("Early stopping triggered at epoch %d (best val_loss=%.4f)", epoch + 1, stopper.best_loss)
            break

    stopper.restore_best(model)
    logger.info("Restored best model (val_loss=%.4f)", stopper.best_loss)
    return model


@torch.no_grad()
def predict_bert(
    model: BertPromptClassifier,
    texts: list[str],
    tokenizer=None,
    batch_size: int = 32,
) -> np.ndarray:
    device = next(model.parameters()).device
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.cfg.model_name)

    all_preds: list[int] = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            padding="max_length",
            max_length=model.cfg.max_seq_len,
            return_tensors="pt",
        )
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)

    return np.array(all_preds)


def save_bert(model: BertPromptClassifier, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg}, path)
    logger.info("Saved BertPromptClassifier to %s", path)


def load_bert(path: str) -> BertPromptClassifier:
    data = torch.load(path, map_location="cpu", weights_only=False)
    model = BertPromptClassifier(data["cfg"])
    model.load_state_dict(data["state_dict"])
    logger.info("Loaded BertPromptClassifier from %s", path)
    return model
