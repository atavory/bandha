"""Per-arm state-table BanditPFN.

This model keeps the public v9 interface but removes learned tensors whose
shape depends on K. Feedback updates a shared recurrent state for the selected
arm, then a permutation-equivariant arm mixer scores all active arms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class BanditPFNPerArm(nn.Module):
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

        self.initial_state = nn.Parameter(torch.zeros(d_model))
        self.event_proj = nn.Sequential(
            nn.Linear(d_ctx + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.update_cell = nn.GRUCell(d_model, d_model)
        self.update_norm = nn.LayerNorm(d_model)

        self.query_proj = nn.Linear(d_ctx, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.arm_mixer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.arm_head = nn.Linear(d_model, 1)

    def _initial_states(
        self,
        batch: int,
        K: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.initial_state.to(device=device, dtype=dtype).view(1, 1, -1).expand(
            batch,
            K,
            -1,
        )

    def _update_states(
        self,
        states: torch.Tensor,
        prev_context: torch.Tensor,
        feedback: torch.Tensor,
    ) -> torch.Tensor:
        """Apply shifted feedback at the current timestep.

        feedback values follow v9's reward-vector convention:
          0 = arm not pulled, 1 = pulled and wrong, 2 = pulled and correct.
        """
        batch, K, d_model = states.shape
        pulled = feedback > 0
        if not bool(pulled.any()):
            return states

        reward = (feedback - 1.0).clamp(min=0.0)
        prev_contexts = prev_context.unsqueeze(1).expand(batch, K, self.d_ctx)
        event_features = torch.cat(
            [
                prev_contexts,
                reward.unsqueeze(-1),
                pulled.to(dtype=states.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        events = self.event_proj(event_features)
        updated = self.update_cell(
            events.reshape(batch * K, d_model),
            states.reshape(batch * K, d_model),
        ).reshape(batch, K, d_model)
        updated = self.update_norm(updated)
        return torch.where(pulled.unsqueeze(-1), updated, states)

    def _score(self, states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(context).unsqueeze(1)
        tokens = states + query

        if not self.activation_checkpointing or not self.training:
            mixed = self.arm_mixer(tokens)
        else:
            mixed = tokens
            for layer in self.arm_mixer.layers:

                def layer_forward(
                    hidden: torch.Tensor, block: nn.Module = layer
                ) -> torch.Tensor:
                    return block(hidden)

                mixed = checkpoint(layer_forward, mixed, use_reentrant=False)
            if self.arm_mixer.norm is not None:
                mixed = self.arm_mixer.norm(mixed)

        return self.arm_head(mixed).squeeze(-1)

    def forward(
        self,
        contexts: torch.Tensor,
        reward_vecs: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced full-sequence forward pass.

        Args:
            contexts:    [batch, T, d_ctx]
            reward_vecs: [batch, T, K] shifted feedback

        Returns:
            logits: [batch, T, K]
        """
        batch, T, _ = contexts.shape
        K = reward_vecs.shape[-1]
        states = self._initial_states(batch, K, contexts.device, contexts.dtype)
        logits = []
        for t in range(T):
            if t > 0:
                states = self._update_states(
                    states,
                    contexts[:, t - 1],
                    reward_vecs[:, t],
                )
            logits.append(self._score(states, contexts[:, t]))
        return torch.stack(logits, dim=1)

    def forward_step(
        self,
        contexts: torch.Tensor,
        reward_vecs: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        return self(contexts[:, : t + 1], reward_vecs[:, : t + 1])[:, t]

    @torch.no_grad()
    def select_arm(
        self,
        contexts: torch.Tensor,
        reward_vecs: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        return self.forward_step(contexts, reward_vecs, t).argmax(dim=-1)
