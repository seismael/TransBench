from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transbench.modules.mixin_modules import GroupedQuerySelfAttentionMixin


class StochasticInductionMixin(nn.Module):
    """Stochastic Induction Layer (SIL) mixin — v2.

    SIL augments a standard token-mixing path (attention) with a parallel
    stochastic "rule" path.  The rule path samples a discrete latent (via
    Gumbel-Softmax) per token and decodes it back into hidden space.

    Key improvements (v2):
    - **Learned gate** controls how much of the induced signal is injected.
      Gate is initialized near zero (bias = −3 → σ ≈ 0.05) so early
      training is dominated by the stable attention path.
    - **Soft Gumbel-Softmax** during training (hard_train=False) reduces
      gradient variance across the 64 latent categories.
    - **Lower default temperature** (0.5) sharpens rule selection sooner.

    Interface contract: `forward(x) -> (B, S, H)`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int | None = None,
        *,
        num_latent_rules: int = 64,
        temperature: float = 0.5,
        hard_train: bool = False,
        hard_eval: bool = True,
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

        self.num_latent_rules = int(num_latent_rules)
        self.temperature = float(temperature)
        self.hard_train = bool(hard_train)
        self.hard_eval = bool(hard_eval)

        self.rule_encoder = nn.Linear(hidden_size, self.num_latent_rules)
        self.rule_decoder = nn.Linear(self.num_latent_rules, hidden_size)
        self.induction_norm = nn.LayerNorm(hidden_size)

        # Learned gate: controls how much induction signal is injected.
        # Scalar per token (hidden → 1) keeps overhead minimal; broadcasts
        # across all hidden dims.  Marked _skip_global_init so
        # archi_modules.init_weight does not reset the bias.
        self.induction_gate = nn.Linear(hidden_size, 1)
        self.induction_gate._skip_global_init = True  # type: ignore[attr-defined]
        nn.init.zeros_(self.induction_gate.weight)
        nn.init.constant_(self.induction_gate.bias, -3.0)

        # Runtime state set by each forward pass.
        self._last_rule_entropy: torch.Tensor | None = None
        self._last_sil_loss: torch.Tensor | None = None

    def set_temperature(self, temperature: float) -> None:
        self.temperature = float(temperature)

    def sil_rule_entropy(self) -> torch.Tensor | None:
        """Mean entropy of the rule distribution (from the last forward)."""
        return self._last_rule_entropy

    def sil_aux_loss(self) -> torch.Tensor | None:
        """Entropy-maximization auxiliary loss (from the last forward).

        Returns *negative* mean entropy so that minimizing this loss
        encourages the rule distribution to stay spread across all latent
        rules, preventing Gumbel-Softmax mode-collapse.

        Returns ``None`` during eval or if forward has not been called.
        """
        return self._last_sil_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Path A: deterministic attention.
        attn_out = self.attention(x)

        # Path B: stochastic induction.
        rule_logits = self.rule_encoder(x)  # (B, S, R)
        tau = max(1e-6, float(self.temperature))
        hard = self.hard_train if self.training else self.hard_eval

        z = F.gumbel_softmax(rule_logits, tau=tau, hard=hard, dim=-1)

        # Compute entropy via fused log_softmax (numerically stable + faster).
        log_probs = F.log_softmax(rule_logits / tau, dim=-1)  # (B, S, R)
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum(dim=-1)  # (B, S)
        mean_ent = ent.mean()

        # Diagnostic: detached entropy for logging.
        self._last_rule_entropy = mean_ent.detach()

        # Aux loss: negative entropy → minimizing pushes towards uniform.
        if self.training:
            self._last_sil_loss = -mean_ent
        else:
            self._last_sil_loss = None

        rule_embedding = self.rule_decoder(z)  # (B, S, H)
        induced = self.induction_norm(rule_embedding)

        # Learned scalar gate: starts near zero, grows as rules become useful.
        alpha = torch.sigmoid(self.induction_gate(x))  # (B, S, 1) broadcasts

        return attn_out + alpha * induced
