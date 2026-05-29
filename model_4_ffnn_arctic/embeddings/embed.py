"""
Generate and cache embeddings using Snowflake Arctic Embed M v2.0.

Arctic Embed M v2.0 uses query/document prompt prefixes for asymmetric tasks,
but for classification we embed all texts the same way (no prefix).
Embeddings are L2-normalized by the model.
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import cfg

logger = logging.getLogger(__name__)


def _cache_path(texts: list[str]) -> Path:
    digest = hashlib.md5("".join(texts[:100]).encode()).hexdigest()[:12]
    return cfg.embeddings_dir / f"embeddings_{len(texts)}_{digest}.npz"


def get_embedder() -> SentenceTransformer:
    logger.info(f"Loading {cfg.embed_model}")
    # Disable memory-efficient attention (requires xformers, unavailable on MPS/CPU)
    model = SentenceTransformer(
        cfg.embed_model,
        trust_remote_code=True,
        config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
    )
    model.max_seq_length = 512
    # PyTorch 2.12 bug: persistent=False position_ids buffer is not properly
    # initialized after weight loading, causing CUDA index-out-of-bounds errors.
    for mod in model.modules():
        if hasattr(mod, "position_ids") and isinstance(mod.position_ids, torch.Tensor):
            mod.register_buffer(
                "position_ids",
                torch.arange(mod.position_ids.shape[0]),
                persistent=False,
            )
    model.eval()
    return model


def embed_texts(
    texts: list[str],
    model: SentenceTransformer | None = None,
    use_cache: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    cache_path = _cache_path(texts)
    if use_cache and cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)["embeddings"]

    if model is None:
        model = get_embedder()

    device = cfg.device
    model = model.to(device)

    logger.info(f"Embedding {len(texts)} texts on {device} (batch={cfg.embed_batch_size})")
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=cfg.embed_batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit-sphere; good for cosine-based classifiers
            device=device,
        )

    if use_cache:
        np.savez_compressed(cache_path, embeddings=embeddings)
        logger.info(f"Cached embeddings to {cache_path}")

    return embeddings  # shape: (N, 768)
