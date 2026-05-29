from dataclasses import dataclass
from typing import Optional

LABEL_SAFE = 0
LABEL_HARMFUL = 1
LABEL_JAILBREAK = 2
ID2LABEL = {0: "safe", 1: "harmful", 2: "jailbreak"}
LABEL2ID = {"safe": 0, "harmful": 1, "jailbreak": 2}
NUM_LABELS = 3


@dataclass
class DataConfig:
    test_size: float = 0.15
    val_size: float = 0.10  # fraction of train+val split
    random_seed: int = 42
    cache_dir: str = "./cache"
    max_samples_per_source: Optional[int] = None


@dataclass
class EmbeddingConfig:
    # Options: "Snowflake/snowflake-arctic-embed-m", "nvidia/NV-Embed-v2"
    model_name: str = "Snowflake/snowflake-arctic-embed-m"
    batch_size: int = 64
    normalize_embeddings: bool = True


@dataclass
class BertConfig:
    model_name: str = "roberta-base"
    frozen: bool = False          # True = frozen encoder, head-only training
    num_labels: int = 3
    head_hidden_size: int = 256   # set 0 for single linear layer (768→3)
    head_dropout: float = 0.1
    max_seq_len: int = 256
    batch_size: int = 16
    lr: float = 2e-5              # use 1e-3 when frozen=True
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    num_epochs: int = 10
    patience: int = 3             # early-stopping patience on val loss
    cv_folds: int = 5
    output_dir: str = "./outputs"
