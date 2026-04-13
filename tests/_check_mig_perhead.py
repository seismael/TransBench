"""Quick check: MIG per-head gate shapes, bias init, both modes."""
from transbench.modules.mig_module import MIGAttention
from transbench.modules.archi_modules import init_weight
import torch

# --- Per-head gate (gate_dim=64) ---
m = MIGAttention(hidden_size=256, num_attention_heads=8, gate_dim=64)
print("per_head:", m.per_head)
print("router:", m.router)
bias = m.router[-1].bias.data.clone()
print(f"gate bias BEFORE init_weight: {bias[:4].tolist()}")

init_weight(m, 0.02)
bias_after = m.router[-1].bias.data
print(f"gate bias AFTER  init_weight: {bias_after[:4].tolist()}")
assert (bias_after == -2.0).all(), f"Bias was reset! Got {bias_after}"
print("✓ Bias preserved at -2.0")

x = torch.randn(2, 16, 256)
out = m(x)
print(f"output shape: {out.shape}")
aux = m.mig_aux_loss().item()
print(f"aux_loss (mean gate): {aux:.4f}  (expect ~0.12 for sigmoid(-2))")
assert 0.05 < aux < 0.25, f"Unexpected aux loss: {aux}"
print("✓ Per-head gate OK")

# --- Simple gate (gate_dim=0) ---
print()
m0 = MIGAttention(hidden_size=256, num_attention_heads=8, gate_dim=0)
print("simple per_head:", m0.per_head)
out0 = m0(x)
print(f"output shape: {out0.shape}")
print("✓ Simple gate OK")

print("\n=== ALL CHECKS PASSED ===")
