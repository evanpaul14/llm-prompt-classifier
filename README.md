# LLM Prompt Safety Classifier

Binary text classifier that intercepts prompts before they reach an LLM and flags them as **safe** (pass through) or **block** (harmful or jailbreak attempt).

| Label | Value | Meaning |
|-------|-------|---------|
| **safe** | 0 | Benign instruction — pass through |
| **block** | 1 | Harmful content or jailbreak attempt — intercept |

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Interactive: choose a model, then type prompts
python predict.py

# One-liner
python predict.py --model 1 --prompt "How do I pick a lock?"
python predict.py --model ffnn_gemma --prompt "Write a poem about autumn"

# Batch from file (one prompt per line)
python predict.py --model 3 --input-file prompts.txt
```

Model options for `--model`: `1` / `tfidf_lr`, `2` / `frozen_bert`, `3` / `roberta`, `4` / `ffnn_gemma`

Models 2–4 are temperature-scaled at inference time: `predict.py` automatically loads
`outputs/<model>/temperature.json` (if present) and divides logits by that temperature
before softmax, so the reported probabilities are calibrated rather than raw logit
confidences. If no `temperature.json` exists for a model, it falls back to `T=1.0`
(uncalibrated).

---

## Project structure

```
llm-prompt-classifier/
├── predict.py                        # unified inference script (all 4 models)
├── requirements.txt
├── data/
│   └── loader.py                     # shared HuggingFace dataset loader + split_dataset()
│
├── model_1_tfidf_lr/
│   ├── model.py                      # TF-IDF + Logistic Regression pipeline
│   └── train.py                      # training script
│
├── model_2_frozen_bert/
│   ├── model.py                      # BertPromptClassifier (frozen RoBERTa encoder)
│   ├── config.py                     # BertConfig dataclass
│   ├── train.py                      # training script
│   └── evaluate.py                   # F1 / FPR / FNR metric helpers
│
├── model_3_roberta_finetune/
│   ├── model.py                      # full fine-tune via AutoModelForSequenceClassification
│   ├── config.py                     # RobertaConfig dataclass
│   ├── train.py                      # training script
│   └── evaluate.py                   # F1 / FPR / FNR metric helpers
│
├── model_4_ffnn_gemma/
│   ├── model.py                      # FFNNClassifier (Linear→BN→GELU→Dropout stack)
│   ├── config.py                     # Config dataclass (all hyperparameters)
│   ├── dataset.py                    # EmbeddingDataset (numpy → torch Dataset)
│   ├── train.py                      # 5-fold CV training script
│   ├── evaluate.py                   # ensemble evaluation with F1 / FPR / FNR
│   ├── embeddings/
│   │   └── embed.py                  # EmbeddingGemma-300m with .npz disk cache
│   └── training/
│       └── cross_val.py              # stratified 5-fold CV trainer
│
├── outputs/                          # trained model artifacts (all models)
│   ├── tfidf_lr/
│   │   └── tfidf_lr.joblib           # Model 1 sklearn pipeline
│   ├── frozen_bert/
│   │   ├── frozen_bert.pt            # Model 2 checkpoint (torch.save)
│   │   └── temperature.json          # Model 2 temperature-scaling calibration
│   ├── roberta_finetuned/            # Model 3 HuggingFace saved model
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── tokenizer*
│   │   └── temperature.json          # Model 3 temperature-scaling calibration
│   └── ffnn_gemma/
│       ├── fold_{1-5}_best.pt        # Model 4 per-fold checkpoints (5-fold ensemble)
│       └── temperature.json          # Model 4 temperature-scaling calibration
│
└── old_models/                       # archived checkpoints (3-class, now superseded)
    ├── tfidf_lr.joblib
    ├── frozen_bert.pt
    ├── roberta_finetuned/
    └── ffnn_arctic_checkpoints/
```

---

## Models

### Model 1 — TF-IDF + Logistic Regression (`model_1_tfidf_lr`)

Classical sklearn pipeline: TF-IDF vectorizer (50k unigram+bigram features,
`sublinear_tf=True`) followed by L-BFGS logistic regression with balanced class
weights. No GPU required; loads and predicts in milliseconds.

**Training:**
```bash
python model_1_tfidf_lr/train.py --max-per-class 10000
```

Key flags: `--max-per-class`, `--max-safe`, `--max-block`, `--C`, `--skip-cv`

---

### Model 2 — Frozen RoBERTa + Classification Head (`model_2_frozen_bert`)

`roberta-base` encoder with **all encoder layers frozen**. Only a two-layer
classification head (`768 → 256 → ReLU → Dropout → 2`) is trained using the
`[CLS]` token representation. Fast to train (head-only); good transformer baseline.

**Training:**
```bash
python model_2_frozen_bert/train.py --lr 1e-3 --batch-size 32 --epochs 10
```

Key flags: `--max-samples`, `--max-safe`, `--batch-size`, `--lr`, `--epochs`, `--patience`

---

### Model 3 — Full Fine-tuned RoBERTa (`model_3_roberta_finetune`)

End-to-end fine-tuning of `roberta-base` via HuggingFace
`AutoModelForSequenceClassification`. All encoder layers + classification head
trained with AdamW, linear LR warmup, and early stopping. Most accurate of the
four models; requires a GPU for practical training time.

**Training:**
```bash
python model_3_roberta_finetune/train.py --lr 2e-5 --batch-size 16 --epochs 10
```

Key flags: `--max-samples`, `--max-safe-samples`, `--batch-size`, `--lr`, `--epochs`, `--patience`

---

### Model 4 — FFNN on EmbeddingGemma-300m (`model_4_ffnn_gemma`)

Two-stage pipeline:

1. **Embedder**: [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
   encodes each prompt to a 768-d L2-normalised vector (frozen, no gradient).
   Embeddings are cached to `.npz` files keyed by model name + MD5 of input texts.

2. **Classifier**: Feedforward network —
   `768 → 512 → BN → GELU → Dropout(0.3) → 256 → BN → GELU → Dropout(0.2) → 128 → BN → GELU → Dropout(0.1) → 2`.
   Trained with AdamW + cosine LR schedule + early stopping (patience 3).

Training runs **5-fold stratified cross-validation**; each fold saves its
best-validation-loss checkpoint. Inference averages logits across all five folds
(ensemble).

> **Note:** `google/embeddinggemma-300m` is a gated model. Accept the licence at
> [hf.co/google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
> and run `huggingface-cli login` before training.

**Training:**
```bash
python model_4_ffnn_gemma/train.py              # embed → 5-fold CV → ensemble eval
python model_4_ffnn_gemma/train.py --skip-cv    # skip CV, evaluate existing checkpoints
python model_4_ffnn_gemma/train.py --limit 200  # quick smoke test
python model_4_ffnn_gemma/train.py --no-cache   # force re-embedding
```

Key flags: `--limit`, `--safe-cap`, `--no-cache`, `--skip-cv`

---

## Metrics reported

All models report the following on the held-out test set:

| Metric | Description |
|--------|-------------|
| **F1 (macro)** | Unweighted mean F1 across both classes |
| **F1 (safe)** | F1 for the safe class |
| **F1 (block)** | F1 for the block class |
| **FPR (safe)** | Fraction of safe prompts wrongly flagged as block |
| **FNR (block)** | Fraction of block prompts missed (passed through) |

Models 2 and 3 also report per-fold CV summaries (mean ± std). Model 4 additionally reports macro-AUC.

---

## Datasets

All models train on the same HuggingFace dataset pool, merged and deduplicated:

| Dataset | HuggingFace ID | Label | Notes |
|---------|---------------|-------|-------|
| JailbreakHub | `walledai/JailbreakHub` | block | ~1.5k confirmed jailbreak prompts |
| Jailbreak Classification | `jackhhao/jailbreak-classification` | block / safe | Community-labelled |
| JailBreakV-28K | `JailbreakV-28K/JailBreakV-28k` | block | 28k jailbreak queries |
| RedTeam-2K | `JailbreakV-28K/JailBreakV-28k` (RedTeam_2K) | block | 2k adversarial questions |
| SALAD-Data | `OpenSafetyLab/Salad-Data` | block | 21k structured harmful questions |
| AdvBench | `walledai/AdvBench` | block | Adversarial harmful behaviour prompts |
| HarmBench (standard) | `walledai/HarmBench` | block | Standardised harmful benchmark |
| HarmBench (contextual) | `walledai/HarmBench` | block | Context-dependent harmful prompts |
| HarmBench (copyright) | `walledai/HarmBench` | block | Copyright-violating prompts |
| LLM-LAT Benign | `LLM-LAT/benign-dataset` | safe | Long-form benign prompts |
| Alpaca | `tatsu-lab/alpaca` | safe | 52k everyday instruction-following prompts |

---

## Re-training

Each model has its own `train.py`. All scripts resolve imports via `sys.path` and can be run from either the repo root or their own directory.

```bash
# From repo root
python model_1_tfidf_lr/train.py
python model_2_frozen_bert/train.py
python model_3_roberta_finetune/train.py
python model_4_ffnn_gemma/train.py
```

---

## Old models

Pre-binary-collapse checkpoints (trained on safe / jailbreak / harmful — 3 classes,
using Snowflake Arctic Embed for model 4) are archived in `old_models/` and are
**not compatible** with the current codebase.
