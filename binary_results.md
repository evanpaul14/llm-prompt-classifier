# Binary Classification Results (safe vs. block)

All models trained on merged HuggingFace datasets. Labels: `0 = safe`, `1 = block`.

---

## Model 1 — TF-IDF + Logistic Regression

**Dataset:** 80,679 total (50,000 safe + 30,679 block — block capped by available data)  
**Split:** 80% dev / 20% held-out test  
**Config:** `max_features=50000`, `ngram_range=(1,2)`, `sublinear_tf=True`, `C=1.0`, `class_weight="balanced"`

### 5-Fold Cross-Validation (dev set)

| Metric | Mean | Std |
|---|---|---|
| F1 macro | 0.9659 | ±0.0009 |
| F1 weighted | 0.9679 | ±0.0008 |
| Accuracy | 0.9678 | ±0.0008 |

### Held-Out Test Set (16,136 samples)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| safe | 0.9817 | 0.9733 | 0.9775 | 10,000 |
| block | 0.9571 | 0.9705 | 0.9637 | 6,136 |
| **macro avg** | **0.9694** | **0.9719** | **0.9706** | 16,136 |
| weighted avg | 0.9724 | 0.9722 | 0.9723 | 16,136 |

| Class | FPR (fall-out) | FNR (miss rate) |
|---|---|---|
| safe | 2.95% | 2.67% |
| block | 2.67% | 2.95% |

---

## Model 2 — Frozen RoBERTa + Classification Head

**Dataset:** 61,358 total (30,679 safe + 30,679 block — balanced; `--max-safe 50000`, block capped supply)  
**Split:** train: 46,018 | val: 6,136 | held-out test: 9,204 (balanced classes throughout)  
**Encoder:** `roberta-base` (frozen); trainable params: 197,378 / 124,843,010 (0.2%)  
**Classifier:** Linear pooler head (roberta-base default)  
**Config:** `max_epochs=10`, `early_stopping_patience=3`, `batch_size≈32`, peak `lr≈1e-3` (linear warmup 6% → linear decay to 0), `class_weight=balanced (1.0/1.0)`  
**CV:** 5-fold on train set; final model retrained on train+val

### 5-Fold Cross-Validation

| Fold | Best val_loss | Best val_acc | F1 macro | F1 safe | F1 block | FNR safe | FNR block |
|---|---|---|---|---|---|---|---|
| 1 | 0.0645 | 0.977 | 0.9771 | 0.9772 | 0.9769 | 1.65% | 2.93% |
| 2 | 0.0608 | 0.979 | 0.9790 | 0.9792 | 0.9788 | 1.36% | 2.84% |
| 3 | 0.0600 | 0.980 | 0.9801 | 0.9802 | 0.9799 | 1.34% | 2.65% |
| 4 | 0.0625 | 0.975 | 0.9753 | 0.9754 | 0.9751 | 1.94% | 3.01% |
| 5 | 0.0621 | 0.979 | 0.9791 | 0.9792 | 0.9790 | 1.55% | 2.63% |
| **mean** | **0.0620 ±0.0016** | **0.978 ±0.002** | **0.9781 ±0.0017** | — | — | **1.57% ±0.22%** | **2.81% ±0.15%** |

### Held-Out Test Set (9,204 samples — 4,602 safe + 4,602 block)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| safe | 0.9740 | 0.9839 | 0.9789 | 4,602 |
| block | 0.9838 | 0.9737 | 0.9787 | 4,602 |
| **macro** | **0.9789** | **0.9788** | **0.9788** | 9,204 |

| Class | FPR | FNR |
|---|---|---|
| safe | 2.63% (121 FAs) | 1.61% (74 missed) |
| block | 1.61% (74 FAs) | 2.63% (121 missed) |

---

## Model 3 — Full RoBERTa Fine-Tune

**Dataset:** 61,358 total (30,679 safe + 30,679 block — balanced; `--max-safe 50000`, block capped supply)  
**Split:** train: 46,018 | val: 6,136 | held-out test: 9,204 (balanced 4,602 safe + 4,602 block throughout)  
**Encoder:** `roberta-base` (full fine-tune); trainable params: 124,647,170 / 124,647,170 (100.0%)  
**Config:** `max_epochs=10`, `early_stopping_patience=3`, `batch_size≈32`, peak `lr≈1.91e-05` (linear warmup 6% → linear decay to 0), `class_weight=balanced (1.0/1.0)`  
**CV:** 5-fold on train set (41,723 train + 10,431 val per fold); final model retrained on train+val (52,154 + 6,136 val)

### 5-Fold Cross-Validation

| Fold | Best val_loss | Best val_acc | F1 macro | F1 safe | F1 block | FNR safe | FNR block |
|---|---|---|---|---|---|---|---|
| 1 | 0.0425 | 0.994 | 0.9937 | 0.9937 | 0.9937 | 0.46% | 0.81% |
| 2 | 0.0354 | 0.994 | 0.9943 | 0.9943 | 0.9944 | 0.92% | 0.21% |
| 3 | 0.0520 | 0.992 | 0.9923 | 0.9923 | 0.9923 | 1.00% | 0.54% |
| 4 | 0.0430 | 0.994 | 0.9940 | 0.9940 | 0.9940 | 0.54% | 0.67% |
| 5 | 0.0397 | 0.993 | 0.9929 | 0.9929 | 0.9929 | 0.75% | 0.67% |
| **mean** | **0.0425 ± 0.0057** | **0.993 ± 0.001** | **0.9934 ± 0.0007** | — | — | **0.73% ± 0.21%** | **0.58% ± 0.20%** |

### Held-Out Test Set (9,204 samples — 4,602 safe + 4,602 block)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| safe | 0.9926 | 0.9935 | 0.9930 | 4,602 |
| block | 0.9935 | 0.9926 | 0.9930 | 4,602 |
| **macro** | **0.9930** | **0.9930** | **0.9930** | 9,204 |

| Class | FPR | FNR |
|---|---|---|
| safe | 0.74% (34 FAs) | 0.65% (30 missed) |
| block | 0.65% (30 FAs) | 0.74% (34 missed) |

---

## Model 4 — FFNN on Frozen Gemma Embeddings

**Dataset:** 80,679 total (50,000 safe + 30,679 block — block capped by available data)  
**Split:** 80% train+val / 20% held-out test  
**Embedder:** `google/embeddinggemma-300m` (frozen, bfloat16, L2-normalized 768-d vectors)  
**Classifier:** Linear(768→512)→BN→GELU→Dropout(0.3) → Linear(512→256)→BN→GELU→Dropout(0.2) → Linear(256→128)→BN→GELU→Dropout(0.1) → Linear(128→2)  
**Config:** `lr=1e-3`, `weight_decay=1e-4`, `batch_size=32`, `max_epochs=50`, `early_stopping_patience=3`, cosine LR scheduler  
**Prediction:** ensemble of 5 fold checkpoints (averaged softmax)

### 5-Fold Cross-Validation

| Fold | Best val_loss | Best val_F1 |
|---|---|---|
| 1 | 0.0209 | 0.9935 |
| 2 | 0.0223 | 0.9923 |
| 3 | 0.0280 | 0.9929 |
| 4 | 0.0258 | 0.9915 |
| 5 | 0.0222 | 0.9939 |
| **mean** | **0.0238 ± 0.0026** | — |

### Held-Out Test Set (16,136 samples — 10,000 safe + 6,136 block, 5-fold ensemble)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| safe | 0.9959 | 0.9932 | 0.9946 | 10,000 |
| block | 0.9889 | 0.9933 | 0.9911 | 6,136 |
| **macro** | **0.9924** | **0.9933** | **0.9928** | 16,136 |

| Metric | Score |
|---|---|
| F1 macro | 0.9928 |
| Accuracy | 0.9933 |
| AUC | 0.9997 |

| Class | FPR | FNR |
|---|---|---|
| safe | 0.68% (68 FAs) | 0.68% (68 missed) |
| block | 0.68% (41 FAs) | 0.67% (41 missed) |

Confusion matrix: [[9932, 68], [41, 6095]]

---

## Summary

| Model | F1 Macro | Accuracy | FPR | FNR |
|---|---|---|---|---|
| 1 — TF-IDF + LR | 0.9706 | 0.9722 | 2.67% | 2.95% |
| 2 — Frozen RoBERTa | 0.9788 | 0.9788 | 1.61% | 2.63% |
| 3 — Full RoBERTa FT | 0.9930 | 0.9930 | 0.65–0.74% | 0.65–0.74% |
| 4 — FFNN + Gemma | 0.9928 | 0.9933 | 0.67–0.68% | 0.67–0.68% |
