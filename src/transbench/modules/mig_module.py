from __future__ import annotations

import torch
import torch.nn as nn

from transbench.modules.mixin_modules import GroupedQuerySelfAttentionMixin


class MIGAttention(nn.Module):
    """Mutual-Information-Gated (MIG) attention.

    Two gate modes selected by ``gate_dim``:

    * **Simple gate** (``gate_dim <= 0``, default):
        g_i = σ(W·x_i + b)          — scalar per token
        out  = Attention(g ⊙ x)
        L_sparse = λ · mean(g)

    * **MLP per-head gate** (``gate_dim > 0``):
        g_i = σ(MLP(x_i))           — [num_heads] per token
        Each attention head receives differently-gated input, enabling
        heads to specialise on different subsets of the context.
        Bias initialised to −2 so gates start near-closed (σ ≈ 0.12);
        the model must actively learn which tokens each head attends to.

    In both modes the gate is applied BEFORE attention and the mean gate
    value is exposed via ``mig_aux_loss()`` for the L1 sparsity penalty.

    **Asymmetric MIG (A-MIG):** When ``layer_keep_ratios`` is provided,
    each layer applies Hard Top-K routing using its own keep ratio.
    Early layers with low ratios (e.g. 0.05) act as aggressive noise
    filters ("Skimmers"), while later layers with ratio 1.0 perform
    standard dense attention on the surviving signal.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int | None = None,
        keep_ratio: float = 0.7,
        layer_keep_ratios: list[float] | None = None,
        router_aux_loss_coef: float = 0.01,
        dropout: float = 0.0,
        bias: bool = True,
        capacity: float | None = None,
        gate_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()

        self.keep_ratio = float(keep_ratio)
        self.layer_keep_ratios = (
            [float(r) for r in layer_keep_ratios] if layer_keep_ratios is not None else None
        )
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        gate_dim = int(gate_dim or 0)
        self.per_head = gate_dim > 0

        if gate_dim > 0:
            # Two-layer MLP gate → per-head gates.
            self.router = nn.Sequential(
                nn.Linear(hidden_size, gate_dim, bias=False),
                nn.SiLU(),
                nn.Linear(gate_dim, num_attention_heads, bias=True),
            )
            # Match model init for weights; preserve negative bias (-2) via
            # _skip_global_init so init_weight() won't reset it to zero.
            final = self.router[-1]
            nn.init.normal_(final.weight, mean=0.0, std=0.02)
            final.bias.data.fill_(-2.0)
            final._skip_global_init = True  # type: ignore[attr-defined]
        else:
            # Simple scalar gate.
            self.router = nn.Linear(hidden_size, 1, bias=True)

        self.attention = GroupedQuerySelfAttentionMixin(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout=float(dropout),
            bias=bool(bias),
        )

        # Injected later by StackedMixinBlock; keep defaults for safety.
        self.layer_idx = 0
        self.layer_id = 0

        self._last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self._last_gate_mean: torch.Tensor | None = None
        self._last_gate_per_token: torch.Tensor | None = None

        # Silence unused kwargs.
        _ = capacity
        _ = kwargs

    def _effective_keep_ratio(self) -> float:
        """Return the keep ratio for the current layer."""
        if self.layer_keep_ratios is not None and self.layer_id < len(self.layer_keep_ratios):
            return self.layer_keep_ratios[self.layer_id]
        return self.keep_ratio

    def mig_aux_loss(self) -> torch.Tensor:
        return self._last_aux_loss

    def _apply_topk_mask(self, gate_scores: torch.Tensor, x: torch.Tensor, ratio: float) -> torch.Tensor:
        """Apply Hard Top-K token selection based on gate scores.

        Args:
            gate_scores: Per-token importance scores ``(B, N)``.
            x: Input tensor ``(B, N, C)`` to mask.
            ratio: Fraction of tokens to keep (0, 1].

        Returns:
            Masked input with non-selected tokens zeroed out.
        """
        B, N, C = x.shape
        k = max(1, int(N * ratio))
        if k >= N:
            return x

        # Select top-k token positions by gate importance.
        _, topk_indices = torch.topk(gate_scores, k, dim=1)  # (B, k)

        # Build binary mask: 1 for kept tokens, 0 for dropped.
        mask = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        mask.scatter_(1, topk_indices, 1.0)  # (B, N)

        return x * mask.unsqueeze(-1)  # (B, N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        ratio = self._effective_keep_ratio()

        if self.per_head:
            # MLP per-head gate: each head gets its own token-importance score.
            gate = torch.sigmoid(self.router(x))           # [B, N, num_heads]
            self._last_aux_loss = gate.mean()
            self._last_gate_mean = gate.mean().detach()
            self._last_gate_per_token = gate.mean(dim=-1).detach()  # [B, N]

            x_heads = x.view(B, N, self.num_attention_heads, self.head_dim)
            x_gated = (x_heads * gate.unsqueeze(-1)).reshape(B, N, C)

            # A-MIG: Hard Top-K based on mean gate across heads.
            if ratio < 1.0:
                x_gated = self._apply_topk_mask(gate.mean(dim=-1), x_gated, ratio)
        else:
            # Simple scalar gate: one importance score per token.
            gate = torch.sigmoid(self.router(x))           # [B, N, 1]
            self._last_aux_loss = gate.mean()
            self._last_gate_mean = gate.mean().detach()
            self._last_gate_per_token = gate.squeeze(-1).detach()  # [B, N]
            x_gated = x * gate

            # A-MIG: Hard Top-K based on scalar gate scores.
            if ratio < 1.0:
                x_gated = self._apply_topk_mask(gate.squeeze(-1), x_gated, ratio)

        out = self.attention(x_gated)
        return out
