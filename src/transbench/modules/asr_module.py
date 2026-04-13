from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transbench.modules.mixin_modules import GroupedQuerySelfAttentionMixin


class ASRAttentionMixin(nn.Module):
    """Attentional Symmetry Regularization (ASR) mixin.

    Enforces that the attention output is invariant to input perturbations
    using a **cosine-similarity** consistency loss (scale-invariant).

    Design:
    - Cosine loss instead of MSE — avoids the magnitude-collapse problem
      where small perturbations produce near-zero MSE.
    - noise_std=0.3 — perturbation must be meaningful relative to signal.
    - detach_target=False — both paths contribute gradients, smoothing the
      loss landscape bidirectionally.

    Contract: `forward(x) -> (B, S, H)`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int | None = None,
        *,
        noise_std: float = 0.3,
        detach_target: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.attention = GroupedQuerySelfAttentionMixin(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout=float(dropout),
            bias=bool(bias),
        )

        self.noise_std = float(noise_std)
        self.detach_target = bool(detach_target)

        self._last_asr_loss: torch.Tensor | None = None

    def set_noise_std(self, noise_std: float) -> None:
        self.noise_std = float(noise_std)

    def asr_aux_loss(self) -> torch.Tensor | None:
        """Returns the last computed ASR auxiliary loss (or None)."""
        return self._last_asr_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_clean = self.attention(x)

        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(x, dtype=torch.float32) * float(self.noise_std)
            x_noisy = x + noise.to(dtype=x.dtype)
            out_noisy = self.attention(x_noisy)

            target = out_clean.detach() if self.detach_target else out_clean

            # Cosine similarity loss: scale-invariant, penalizes directional
            # divergence between clean and noisy outputs.
            cos_sim = F.cosine_similarity(target, out_noisy, dim=-1)  # [B, S]
            self._last_asr_loss = (1.0 - cos_sim).mean()
        else:
            self._last_asr_loss = None

        return out_clean
