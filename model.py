"""BanditPFN — causal Transformer for contextual bandits.

Two forward paths:
  forward():      teacher-forced, full-sequence, one pass with causal mask.
                   Used during training on logged histories.
  forward_step(): autoregressive, one step at a time.
                   Used during online inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class BanditPFN(nn.Module):
    def __init__(
        self,
        d_ctx: int = 10,
        K: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_T: int = 1024,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_ctx = d_ctx
        self.K = K
        self.d_model = d_model
        self.max_T = max_T
        self.activation_checkpointing = activation_checkpointing

        self.input_proj = nn.Linear(d_ctx + K, d_model)
        self.register_buffer("pos_enc", self._sinusoidal_pe(max_T, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.arm_head = nn.Linear(d_model, K)

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        _, T, _ = tokens.shape
        x = self.input_proj(tokens) + self.pos_enc[:, :T]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        if not self.activation_checkpointing or not self.training:
            return self.transformer(x, mask=mask, is_causal=True)

        # Checkpoint encoder layers individually to reduce activation memory.
        for layer in self.transformer.layers:

            def layer_forward(
                hidden: torch.Tensor, block: nn.Module = layer
            ) -> torch.Tensor:
                return block(hidden, src_mask=mask, is_causal=True)

            x = checkpoint(layer_forward, x, use_reentrant=False)

        if self.transformer.norm is not None:
            x = self.transformer.norm(x)
        return x

    @staticmethod
    def _sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(length, d_model)
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(
        self,
        contexts: torch.Tensor,
        reward_vecs: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced full-sequence forward pass.

        Args:
            contexts:    [batch, T, d_ctx]
            reward_vecs: [batch, T, K] — shifted feedback

        Returns:
            logits: [batch, T, K]
        """
        tokens = torch.cat([contexts, reward_vecs], dim=-1)
        return self.arm_head(self._encode(tokens))

    def forward_step(
        self,
        contexts: torch.Tensor,
        reward_vecs: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Autoregressive forward pass up to position t.

        Args:
            contexts:    [batch, T, d_ctx]
            reward_vecs: [batch, T, K]
            t: current timestep

        Returns:
            logits: [batch, K]
        """
        tokens = torch.cat([contexts[:, : t + 1], reward_vecs[:, : t + 1]], dim=-1)
        return self.arm_head(self._encode(tokens)[:, t])

    @torch.no_grad()
    def select_arm(
        self, contexts: torch.Tensor, reward_vecs: torch.Tensor, t: int
    ) -> torch.Tensor:
        return self.forward_step(contexts, reward_vecs, t).argmax(dim=-1)
