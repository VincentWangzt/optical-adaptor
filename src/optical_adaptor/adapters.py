from __future__ import annotations

import torch
from torch import nn


class MLPAdapter(nn.Module):
    """Token-wise projector; the image sequence length is preserved."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [images, tokens, {self.input_dim}] embeddings")
        return self.projection(embeddings)
