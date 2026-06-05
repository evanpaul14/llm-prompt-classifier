from dataclasses import dataclass
from typing import Optional

LABEL_SAFE = 0
LABEL_JAILBREAK = 1
LABEL_HARMFUL = 2
ID2LABEL = {0: "safe", 1: "jailbreak", 2: "harmful"}
LABEL2ID = {"safe": 0, "jailbreak": 1, "harmful": 2}
NUM_LABELS = 3


@dataclass
class DataConfig:
    test_size: float = 0.15
    val_size: float = 0.10  # fraction of train+val split
    random_seed: int = 42
    cache_dir: str = "./cache"
    max_samples_per_source: Optional[int] = None


@dataclass
class RobertaConfig:
    model_name: str = "roberta-base"
    num_labels: int = 3
    max_seq_len: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    num_epochs: int = 10
    patience: int = 3
    cv_folds: int = 5
    output_dir: str = "./outputs"
