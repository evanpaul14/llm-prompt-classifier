# LLM Prompt Safety Classifier

Four models that classify user prompts into three safety categories:

| Label | Meaning |
|-------|---------|
| **safe** (0) | Benign instruction — pass through |
| **jailbreak** (1) | Attempt to bypass model safety guidelines |
| **harmful** (2) | Directly requests dangerous / policy-violating content |

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Interactive: choose a model, then type prompts
python predict.py

# One-liner
python predict.py --model 1 --prompt "How do I pick a lock?"
python predict.py --model ffnn_arctic --prompt "Write a poem about autumn"

# Batch from file (one prompt per line)
python predict.py --model 3 --input-file prompts.txt
```

Model options for `--model`: `1` / `tfidf_lr`, `2` / `frozen_bert`, `3` / `roberta`, `4` / `ffnn_arctic`

---

## Project structure

```
llm-prompt-classifier/
├── predict.py                        # unified inference script (all 4 models)
├── requirements.txt
├── data/
│   └── loader.py                     # shared HuggingFace dataset loader
│
├── model_1_tfidf_lr/
│   ├── model.py                      # TF-IDF + Logistic Regression pipeline
│   ├── train.py                      # training script
│   └── tfidf_lr.joblib               # ← saved model
│
├── model_2_frozen_bert/
│   ├── model.py                      # BertPromptClassifier (frozen encoder)
│   ├── config.py                     # BertConfig dataclass
│   ├── train.py                      # training script
│   ├── evaluate.py                   # metric helpers
│   └── frozen_bert.pt                # ← saved model
│
├── model_3_roberta_finetune/
│   ├── model.py                      # full fine-tune via AutoModelForSequenceClassification
│   ├── config.py                     # RobertaConfig dataclass
│   ├── train.py                      # training script
│   ├── evaluate.py                   # metric helpers
│   └── roberta_finetuned/            # ← saved model (HuggingFace format)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
└── model_4_ffnn_arctic/
    ├── model.py                      # FFNNClassifier (Linear→BN→GELU→Dropout stack)
    ├── config.py                     # Config dataclass (all hyperparameters)
    ├── train.py                      # 5-fold CV training script
    ├── evaluate.py                   # ensemble evaluation
    ├── embeddings/
    │   └── embed.py                  # Arctic Embed M v2.0 with .npz disk cache
    └── checkpoints/                  # ← saved models (one per CV fold)
        ├── fold_1_best.pt
        ├── fold_2_best.pt
        ├── fold_3_best.pt
        ├── fold_4_best.pt
        └── fold_5_best.pt
```

---

## Models

### Model 1 — TF-IDF + Logistic Regression (`model_1_tfidf_lr`)

A classical sklearn pipeline: TF-IDF vectorizer (up to 50k unigram+bigram features,
`sublinear_tf=True`) followed by an L-BFGS logistic regression with balanced class
weights. Lightest model; no GPU required; loads and predicts in milliseconds.

**Saved file:** `tfidf_lr.joblib` (sklearn Pipeline, loadable with `joblib.load`)

**Training:**
```bash
cd model_1_tfidf_lr
python train.py --max-per-class 10000 --output tfidf_lr.joblib
```

---

### Model 2 — Frozen RoBERTa + Classification Head (`model_2_frozen_bert`)

`roberta-base` encoder with **all encoder layers frozen**. Only a two-layer
classification head (`768 → 256 → ReLU → Dropout → 3`) is trained. Uses
`[CLS]` token representation. Fast to train (head-only), good baseline for
transformer-based classification.

**Saved file:** `frozen_bert.pt` (PyTorch state dict + config via `torch.save`)

**Training:**
```bash
cd model_2_frozen_bert
python train.py --lr 1e-3 --batch-size 32 --epochs 10
```

---

### Model 3 — Full Fine-tuned RoBERTa (`model_3_roberta_finetune`)

End-to-end fine-tuning of `roberta-base` using HuggingFace
`AutoModelForSequenceClassification`. All encoder layers + the built-in
classification head are updated with AdamW + linear LR warmup + early stopping.
Most powerful of the four models; requires a GPU for reasonable training time.

**Saved directory:** `roberta_finetuned/` (HuggingFace `save_pretrained` format —
loadable with `AutoModelForSequenceClassification.from_pretrained`)

**Training:**
```bash
cd model_3_roberta_finetune
python train.py --lr 2e-5 --batch-size 16 --epochs 10
```

---

### Model 4 — FFNN on Snowflake Arctic Embeddings (`model_4_ffnn_arctic`)

A two-stage pipeline:

1. **Embedder**: [Snowflake Arctic Embed M v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0)
   encodes each prompt to a 768-d L2-normalised vector (frozen, no gradient).
   Embeddings are cached to `.npz` files keyed by an MD5 of the input texts.

2. **Classifier**: Small feedforward network — `768 → 512 → BN → GELU → Dropout(0.3)
   → 256 → BN → GELU → Dropout(0.2) → 128 → BN → GELU → Dropout(0.1) → 3`.
   Trained with AdamW + cosine LR schedule + early stopping.

Training uses **5-fold stratified cross-validation**; each fold saves its
best-validation-loss checkpoint. Inference averages logits across all five fold
models (ensemble).

**Saved files:** `checkpoints/fold_{1..5}_best.pt` (PyTorch state dicts)

**Training:**
```bash
cd model_4_ffnn_arctic
python train.py              # full pipeline: embed → 5-fold CV → ensemble eval
python train.py --skip-cv    # skip CV, just evaluate existing checkpoints
python train.py --limit 200  # quick smoke test
```

---

## Datasets

All models are trained on the same pool of HuggingFace datasets, merged and
deduplicated across sources:

| Dataset | HuggingFace ID | Label | Description |
|---------|---------------|-------|-------------|
| JailbreakHub | `walledai/JailbreakHub` | jailbreak | ~1.5k confirmed jailbreak system prompts |
| Jailbreak Classification | `jackhhao/jailbreak-classification` | jailbreak / safe | Community-labelled jailbreak + benign prompts |
| JailBreakV-28K | `JailbreakV-28K/JailBreakV-28k` | jailbreak | 28k multimodal jailbreak queries (text subset) |
| RedTeam-2K | `JailbreakV-28K/JailBreakV-28k` (RedTeam_2K) | jailbreak | 2k red-team adversarial questions |
| SALAD-Data | `OpenSafetyLab/Salad-Data` | harmful | 21k structured harmful questions |
| AdvBench | `walledai/AdvBench` | harmful | Adversarial harmful behaviour prompts |
| HarmBench (standard) | `walledai/HarmBench` | harmful | Standardised harmful behaviour benchmark |
| HarmBench (contextual) | `walledai/HarmBench` | harmful | Context-dependent harmful prompts |
| HarmBench (copyright) | `walledai/HarmBench` | harmful | Copyright-violating prompt benchmark |
| LLM-LAT Benign | `LLM-LAT/benign-dataset` | safe | Long-form benign user prompts |
| Alpaca | `tatsu-lab/alpaca` | safe | 52k short everyday instruction-following prompts |

Data loading is handled by `data/loader.py`, which downloads, filters, deduplicates
(within-source and cross-source), and optionally balances class counts before
train/val/test splitting.

---

## Re-training any model

Each model directory contains its own `train.py`. All training scripts load data
via `data/loader.py` and expect to be run from the repo root or their own directory
(imports adjust via `sys.path`).

```bash
# From repo root — example for model 1
python model_1_tfidf_lr/train.py --output model_1_tfidf_lr/tfidf_lr.joblib
```
