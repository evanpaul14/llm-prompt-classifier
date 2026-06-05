from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # Paths
    cache_dir: Path = Path("cache")
    embeddings_dir: Path = Path("cache/embeddings")
    checkpoints_dir: Path = Path("checkpoints")

    # Embedding model
    embed_model: str = "google/embeddinggemma-300m"
    embed_dim: int = 768          # embeddinggemma-300m native output dim
    embed_batch_size: int = 64

    # FFNN architecture
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    dropout_rates: list = field(default_factory=lambda: [0.3, 0.2, 0.1])

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    early_stopping_patience: int = 3
    lr_scheduler: str = "cosine"  # "cosine" or "plateau"

    # Cross-validation
    n_folds: int = 5

    # Data
    test_size: float = 0.2
    random_seed: int = 42
    max_text_length: int = 512  # chars, not tokens

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # macOS spawn-based multiprocessing causes DataLoader worker loops; use 0 workers
    num_workers: int = 0 if torch.backends.mps.is_available() else 4
    pin_memory: bool = torch.cuda.is_available()  # only useful for CUDA

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.embeddings_dir = Path(self.embeddings_dir)
        self.checkpoints_dir = Path(self.checkpoints_dir)
        for d in [self.cache_dir, self.embeddings_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()
