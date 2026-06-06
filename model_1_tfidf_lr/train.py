import argparse
import os
import sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.loader import load_all, SAFE, BLOCK
from model import build_pipeline


def run_cv(model, X, y, n_folds=5):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scoring = ["f1_macro", "f1_weighted", "accuracy"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    print(f"\n{n_folds}-Fold Cross-Validation (train set only):")
    print(f"  {'Metric':<18} {'Mean':>8}  {'Std':>8}  Folds")
    for metric in scoring:
        scores = results[f"test_{metric}"]
        fold_str = "  ".join(f"{s:.4f}" for s in scores)
        print(f"  {metric:<18} {scores.mean():>8.4f}  {scores.std():>8.4f}  [{fold_str}]")


def print_error_rates(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(f"  {'Class':<12} {'FPR (fall-out)':>16}  {'FNR (miss rate)':>16}")
    print(f"  {'-'*12}  {'-'*16}  {'-'*16}")
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp          # true class predicted as other
        fp = cm[:, i].sum() - tp          # other classes predicted as this
        tn = cm.sum() - tp - fn - fp
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        print(f"  {cls:<12} {fpr:>16.4f}  {fnr:>16.4f}")
    print()


def main(args):
    if args.no_salad:
        import logging
        logging.warning("--no-salad is not supported by the current data loader and will be ignored.")

    per_class = {
        SAFE: args.max_safe,
        BLOCK: args.max_block,
    }
    if any(v is not None for v in per_class.values()):
        max_samples_by_label = {k: (v if v is not None else args.max_per_class) for k, v in per_class.items()}
    else:
        max_samples_by_label = {SAFE: args.max_per_class, BLOCK: args.max_per_class}

    df = load_all(balance=False, max_samples_by_label=max_samples_by_label)

    print(f"\nDataset summary ({len(df):,} total):")
    print(df["label"].value_counts().to_string())

    # Hold out test set
    X_dev, X_test, y_dev, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )
    print(f"\nDev: {len(X_dev):,}  Test (held-out): {len(X_test):,}")

    model = build_pipeline(C=args.C)

    if not args.skip_cv:
        run_cv(build_pipeline(C=args.C), X_dev, y_dev, n_folds=5)

    # Final model: fit on full dev set, evaluate on held-out test set once
    print("\nFitting final model on full dev set...")
    model.fit(X_dev, y_dev)

    print("\n--- Final held-out test set evaluation ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    print_error_rates(y_test, y_pred, model.classes_)

    joblib.dump(model, args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF + LR prompt classifier")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=10_000,
        help="Cap all classes at this many examples (default: 10000); overridden per-class by --max-jailbreak/harmful/safe",
    )
    parser.add_argument(
        "--max-block",
        type=int,
        default=None,
        help="Cap block class (overrides --max-per-class for this class)",
    )
    parser.add_argument(
        "--max-safe",
        type=int,
        default=None,
        help="Cap safe class (overrides --max-per-class for this class)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Logistic regression regularization strength (default: 1.0)",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation and go straight to final training",
    )
    parser.add_argument(
        "--no-salad",
        action="store_true",
        help="Exclude OpenSafetyLab/Salad-Data from training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/tfidf_lr/tfidf_lr.joblib",
        help="Path to save the trained model",
    )
    main(parser.parse_args())
