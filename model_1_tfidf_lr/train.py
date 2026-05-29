import argparse
import os
import sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_all
from models.tfidf_lr import build_pipeline


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
    from data.loader import LOADERS, JAILBREAK, HARMFUL, SAFE
    sources = [s for s in LOADERS if not (args.no_salad and s == "salad")]

    per_class = {
        JAILBREAK: args.max_jailbreak,
        HARMFUL: args.max_harmful,
        SAFE: args.max_safe,
    }
    # Use per-class dict if any were set explicitly, else fall back to global flag.
    if any(v is not None for v in per_class.values()):
        # Fill unset per-class caps from --max-per-class.
        caps = {k: (v if v is not None else args.max_per_class) for k, v in per_class.items()}
        max_per_class = caps
    else:
        max_per_class = args.max_per_class

    df, unused = load_all(sources=sources, max_per_class=max_per_class, return_unused=True)

    print(f"\nDataset summary ({len(df):,} total):")
    print(df["label"].value_counts().to_string())

    if not unused.empty:
        print(f"\nExcluded by cap ({len(unused):,} total) — sample prompts not used in training:")
        for label in sorted(unused["label"].unique()):
            subset = unused[unused["label"] == label]
            samples = subset["text"].sample(min(5, len(subset)), random_state=42)
            print(f"\n  [{label.upper()}]  ({len(subset):,} excluded)")
            for text in samples:
                snippet = text.replace("\n", " ")
                print(f"    - {snippet}")

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
        "--max-jailbreak",
        type=int,
        default=None,
        help="Cap jailbreak class (overrides --max-per-class for this class)",
    )
    parser.add_argument(
        "--max-harmful",
        type=int,
        default=None,
        help="Cap harmful class (overrides --max-per-class for this class)",
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
        default="tfidf_lr.joblib",
        help="Path to save the trained model (default: tfidf_lr.joblib)",
    )
    main(parser.parse_args())
