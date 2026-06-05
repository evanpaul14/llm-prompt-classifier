"""
Load and merge all datasets into a single (text, label) DataFrame.

Label convention:
    0 = safe      (pass through)
    1 = jailbreak (redirect / warn)
    2 = harmful   (block)
"""

import logging
import re
from typing import Optional, Union

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split as _tts

logger = logging.getLogger(__name__)

SAFE        = 0
BLOCK       = 1   # covers both jailbreak and harmful
CLASS_NAMES = ["safe", "block"]

_TEXT_CANDIDATES = [
    "prompt", "text", "question", "behavior", "Behavior",
    "goal", "jailbreak_query", "query", "content", "input", "instruction",
]

_SOURCES: list[dict] = [
    # ── walledai/JailbreakHub ─────────────────────────────────────────────
    # Only the confirmed jailbreak half; the "safe" half is failed jailbreak
    # system prompts (persona injections, token flooding, etc.) — not real
    # safe user queries, so it is excluded.
    dict(dataset_id="walledai/JailbreakHub", subset=None, split="train",
         text_col="prompt", label=BLOCK, filter_col="jailbreak", filter_val=True,
         source="JailbreakHub_jailbreak"),

    # ── jackhhao/jailbreak-classification ────────────────────────────────
    dict(dataset_id="jackhhao/jailbreak-classification", subset=None, split="train",
         text_col="prompt", label=BLOCK, filter_col="type", filter_val="jailbreak",
         source="jackhhao_jailbreak_train"),
    dict(dataset_id="jackhhao/jailbreak-classification", subset=None, split="test",
         text_col="prompt", label=BLOCK, filter_col="type", filter_val="jailbreak",
         source="jackhhao_jailbreak_test"),
    dict(dataset_id="jackhhao/jailbreak-classification", subset=None, split="train",
         text_col="prompt", label=SAFE, filter_col="type", filter_val="benign",
         source="jackhhao_safe_train"),
    dict(dataset_id="jackhhao/jailbreak-classification", subset=None, split="test",
         text_col="prompt", label=SAFE, filter_col="type", filter_val="benign",
         source="jackhhao_safe_test"),

    # ── JailbreakV-28K/JailBreakV-28k ────────────────────────────────────
    dict(dataset_id="JailbreakV-28K/JailBreakV-28k", subset="JailBreakV_28K",
         split="JailBreakV_28K", text_col="jailbreak_query", label=BLOCK,
         source="JailBreakV28K"),
    dict(dataset_id="JailbreakV-28K/JailBreakV-28k", subset="RedTeam_2K",
         split="RedTeam_2K", text_col="question", label=BLOCK,
         source="RedTeam2K"),

    # ── OpenSafetyLab/Salad-Data ──────────────────────────────────────────
    dict(dataset_id="OpenSafetyLab/Salad-Data", subset="base_set", split="train",
         text_col="question", label=BLOCK, source="SaladData"),

    # ── walledai/AdvBench ─────────────────────────────────────────────────
    dict(dataset_id="walledai/AdvBench", subset=None, split="train",
         text_col="prompt", label=BLOCK, source="AdvBench"),

    # ── walledai/HarmBench ────────────────────────────────────────────────
    dict(dataset_id="walledai/HarmBench", subset="standard", split="train",
         text_col="prompt", label=BLOCK, source="HarmBench_standard"),
    dict(dataset_id="walledai/HarmBench", subset="contextual", split="train",
         text_col="prompt", label=BLOCK, source="HarmBench_contextual"),
    dict(dataset_id="walledai/HarmBench", subset="copyright", split="train",
         text_col="prompt", label=BLOCK, source="HarmBench_copyright"),

    # ── LLM-LAT/benign-dataset ────────────────────────────────────────────
    dict(dataset_id="LLM-LAT/benign-dataset", subset=None, split="train",
         text_col="prompt", label=SAFE, source="benign"),

    # ── tatsu-lab/alpaca ──────────────────────────────────────────────────
    # 52k short everyday instructions (avg 60 chars) — fills the gap left by
    # LLM-LAT/benign being mostly long verbose prompts.
    dict(dataset_id="tatsu-lab/alpaca", subset=None, split="train",
         text_col="instruction", label=SAFE, source="alpaca"),
]


def _clean(text: str, max_len: int = 512) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()[:max_len]


def _find_text_column(column_names: list[str], preferred: Optional[str]) -> str:
    if preferred and preferred in column_names:
        return preferred
    for cand in _TEXT_CANDIDATES:
        if cand in column_names:
            logger.debug("Auto-detected text column %r from %s", cand, column_names)
            return cand
    raise ValueError(
        f"Cannot find a text column in {column_names}. "
        "Add the correct name to _TEXT_CANDIDATES or set text_col explicitly."
    )


def _load_source(
    dataset_id: str,
    label: int,
    source: str,
    subset: Optional[str] = None,
    split: str = "train",
    text_col: Optional[str] = None,
    filter_col: Optional[str] = None,
    filter_val: Optional[Union[str, bool]] = None,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    try:
        ds = load_dataset(dataset_id, subset, split=split, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Skipping %s/%s (%s): %s", dataset_id, subset, split, exc)
        return pd.DataFrame(columns=["text", "label", "source"])

    col = _find_text_column(ds.column_names, text_col)

    if filter_col is not None and filter_val is not None:
        ds = ds.filter(lambda row: row[filter_col] == filter_val)

    texts = ds[col]
    if max_samples:
        texts = texts[:max_samples]

    df = pd.DataFrame({"text": texts, "label": label, "source": source})
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].apply(_clean)
    df = df[df["text"].str.len() > 10]

    logger.info("%-35s  label=%d  %5d rows", source, label, len(df))
    return df


def load_all(
    balance: bool = True,
    random_seed: int = 42,
    max_per_source: Optional[int] = None,
    max_samples_by_label: Optional[dict] = None,
) -> pd.DataFrame:
    """Return a deduplicated, shuffled DataFrame with columns [text, label, source].

    max_samples_by_label caps specific classes after loading, e.g. {0: 50000} loads
    all jailbreak/harmful samples but limits safe to 50k. Overrides balance for capped
    classes — set balance=False when using this to avoid further downsampling.
    """
    frames = [_load_source(**src, max_samples=max_per_source) for src in _SOURCES]

    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="text")
            .sample(frac=1, random_state=random_seed)
            .reset_index(drop=True))

    if max_samples_by_label:
        parts = []
        for lbl, cap in max_samples_by_label.items():
            parts.append(df[df["label"] == lbl].head(cap))
        rest = df[~df["label"].isin(max_samples_by_label)]
        df = (pd.concat([rest] + parts, ignore_index=True)
                .sample(frac=1, random_state=random_seed)
                .reset_index(drop=True))

    if balance:
        n_min = df["label"].value_counts().min()
        parts = [
            grp.sample(n=n_min, random_state=random_seed)
            for _, grp in df.groupby("label")
        ]
        df = (pd.concat(parts, ignore_index=True)
                .sample(frac=1, random_state=random_seed)
                .reset_index(drop=True))

    for i, name in enumerate(CLASS_NAMES):
        logger.info("  %s: %d", name, (df["label"] == i).sum())
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.10,
    random_seed: int = 42,
) -> tuple:
    """Split into (train, val, test) DataFrames, stratified by label.

    val_size is the desired fraction of the *total* dataset (not of train+val).
    """
    trainval, test = _tts(
        df, test_size=test_size, stratify=df["label"], random_state=random_seed
    )
    val_frac = val_size / (1.0 - test_size)
    train, val = _tts(
        trainval, test_size=val_frac, stratify=trainval["label"], random_state=random_seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
