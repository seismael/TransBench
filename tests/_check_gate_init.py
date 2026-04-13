"""Quick check that SIL gate bias survives init_weight."""
import torch
from transbench.modules.sil_module import StochasticInductionMixin
from transbench.modules.archi_modules import StackedMixinForCausalLM
from transbench.modules.ffn_modules import FFN

mixin = StochasticInductionMixin(
    hidden_size=256, num_attention_heads=8,
    num_latent_rules=64, temperature=0.5, hard_train=False,
)
ffn = FFN(hidden_size=256, intermediate_size=1024)

print("BEFORE model construction:")
print(f"  gate bias = {mixin.induction_gate.bias.data[0]:.4f}")
print(f"  gate weight norm = {mixin.induction_gate.weight.data.norm():.6f}")

model = StackedMixinForCausalLM(
    num_layers=8, hidden_size=256, initializer_range=0.02,
    mixin_module=mixin, ffn_module=ffn,
    freeze_lm_modules=False, vocab_size=50257,
)

print("\nAFTER model construction (init_weight applied):")
for i, layer in enumerate(model.stacked_mixin_block.layers):
    inner = layer.mixin_module.module
    gate = inner.induction_gate
    b = gate.bias.data[0].item()
    w = gate.weight.data.norm().item()
    ok = "✓" if abs(b - (-3.0)) < 0.01 else "✗ BUG"
    print(f"  Layer {i}: gate bias={b:.4f} {ok}, weight_norm={w:.6f}")

# Verify gate output is near σ(-3) ≈ 0.05
x = torch.randn(1, 8, 256)
layer0 = model.stacked_mixin_block.layers[0].mixin_module.module
alpha = torch.sigmoid(layer0.induction_gate(x))
print(f"\nGate output mean: {alpha.mean().item():.4f} (expect ~0.05)")
assert abs(alpha.mean().item() - 0.05) < 0.03, "Gate not near 0.05!"
print("GATE INIT CHECK PASSED ✓")
