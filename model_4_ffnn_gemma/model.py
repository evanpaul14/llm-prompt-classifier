"""
Feedforward classifier on top of frozen embeddings.

Architecture: Linear → BN → GELU → Dropout, repeated per hidden layer,
then Linear(last_dim → NUM_CLASSES).  CrossEntropyLoss handles softmax.
"""

import torch
import torch.nn as nn
from config import cfg

NUM_CLASSES = 2  # safe / block


class FFNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = cfg.embed_dim,
        hidden_dims: list[int] = cfg.hidden_dims,
        dropout_rates: list[float] = cfg.dropout_rates,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        assert len(hidden_dims) == len(dropout_rates)

        layers = []
        prev_dim = input_dim
        for h_dim, drop in zip(hidden_dims, dropout_rates):
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(drop),
            ]
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 2) — raw logits
