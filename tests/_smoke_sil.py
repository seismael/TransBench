"""Quick smoke test for SIL aux loss."""
import torch
from transbench.modules.sil_module import StochasticInductionMixin

m = StochasticInductionMixin(hidden_size=64, num_attention_heads=4)
m.train()
x = torch.randn(1, 8, 64)
out = m(x)
loss = m.sil_aux_loss()
print(f"output shape: {out.shape}")
print(f"sil_aux_loss: {loss}")
print(f"sil_rule_entropy: {m.sil_rule_entropy()}")
print(f"has_grad: {loss.requires_grad}")
assert loss is not None, "sil_aux_loss should not be None during training"
assert loss.requires_grad, "sil_aux_loss should carry gradients"
assert loss.item() < 0.0, "negative entropy should be negative"

# Verify eval mode returns None
m.eval()
_ = m(x)
assert m.sil_aux_loss() is None, "sil_aux_loss should be None in eval"

print("ALL SMOKE TESTS PASSED")
