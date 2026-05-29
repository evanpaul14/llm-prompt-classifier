"""Evaluation utilities: per-class F1, macro F1, and per-class FNR/FPR."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)

_LABELS = [0, 1, 2]
_LABEL_NAMES = ["safe", "harmful", "jailbreak"]


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> dict[str, float]:
    """Return macro F1, per-class F1, and per-class FNR/FPR."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=_LABELS)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0, labels=_LABELS)
    cm = confusion_matrix(y_true, y_pred, labels=_LABELS)

    metrics: dict[str, float] = {"f1_macro": float(f1_macro)}
    for i, name in enumerate(_LABEL_NAMES):
        metrics[f"f1_{name}"] = float(f1_per[i])

    # Per-class FNR and FPR using one-vs-rest.
    for i, name in enumerate(_LABEL_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp          # true positives missed
        fp = cm[:, i].sum() - tp          # other classes predicted as this class
        tn = cm.sum() - tp - fn - fp

        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics[f"fnr_{name}"] = float(fnr)
        metrics[f"fpr_{name}"] = float(fpr)
        metrics[f"tp_{name}"] = int(tp)
        metrics[f"fn_{name}"] = int(fn)
        metrics[f"fp_{name}"] = int(fp)
        metrics[f"tn_{name}"] = int(tn)

    return metrics


def print_metrics(metrics: dict[str, float], title: str = "") -> None:
    if title:
        print(f"\n{'─' * 56}")
        print(f"  {title}")
        print(f"{'─' * 56}")
    print(f"  F1 (macro)      : {metrics['f1_macro']:.4f}")
    for name in _LABEL_NAMES:
        f1 = metrics[f"f1_{name}"]
        fnr = metrics[f"fnr_{name}"]
        fpr = metrics[f"fpr_{name}"]
        fn = metrics[f"fn_{name}"]
        fp = metrics[f"fp_{name}"]
        print(
            f"  F1 ({name:<9}): {f1:.4f}  "
            f"FNR={fnr:.4f} ({fn} missed)  "
            f"FPR={fpr:.4f} ({fp} false alarms)"
        )


def aggregate_cv_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean ± std across CV folds."""
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], float)]
    agg: dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg


def print_cv_metrics(agg: dict[str, float]) -> None:
    print("\n── Cross-validation summary ──────────────────────────────")
    stats = ["f1_macro"] + [f"f1_{n}" for n in _LABEL_NAMES] + [f"fnr_{n}" for n in _LABEL_NAMES]
    for stat in stats:
        mean = agg.get(f"{stat}_mean", float("nan"))
        std = agg.get(f"{stat}_std", float("nan"))
        print(f"  {stat:<20}: {mean:.4f} ± {std:.4f}")
