"""
Generate and cache embeddings using google/embeddinggemma-300m.

EmbeddingGemma does not support float16; we use bfloat16 on GPU/MPS and
float32 on CPU. For classification we embed all texts the same way (no
asymmetric query/document prompts). Embeddings are L2-normalised by the model.
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
    model_slug = cfg.embed_model.replace("/", "_")
    digest = hashlib.md5("".join(texts[:100]).encode()).hexdigest()[:12]
    return cfg.embeddings_dir / f"embeddings_{model_slug}_{len(texts)}_{digest}.npz"


def get_embedder() -> SentenceTransformer:
    logger.info(f"Loading {cfg.embed_model}")
    # float16 is unsupported by EmbeddingGemma; use bfloat16 on accelerators
    dtype = torch.bfloat16 if cfg.device != "cpu" else torch.float32
    model = SentenceTransformer(
        cfg.embed_model,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": dtype},
    )
    model.max_seq_length = 512
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
            normalize_embeddings=True,
            device=device,
        )

    if use_cache:
        np.savez_compressed(cache_path, embeddings=embeddings)
        logger.info(f"Cached embeddings to {cache_path}")

    return embeddings  # shape: (N, 768)
