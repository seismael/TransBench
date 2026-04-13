"""Check SIL gate evolution and MIG gate values during training."""
import json, glob, torch, os, sys
sys.path.insert(0, "src")

from transbench.modules.sil_module import StochasticInductionMixin
from transbench.modules.mig_module import MIGAttention

# Check SIL gate initialization
sil = StochasticInductionMixin(hidden_size=128, num_attention_heads=8, num_latent_rules=32, temperature=0.5, hard_train=False)
gate_bias = sil.induction_gate.bias.item()
gate_weight = sil.induction_gate.weight.data
print(f"SIL gate init: bias={gate_bias:.4f}, sigmoid(bias)={torch.sigmoid(torch.tensor(gate_bias)).item():.4f}")
print(f"  Gate weight: all zeros = {(gate_weight == 0).all().item()}")
print(f"  => Initial gate output: ~{torch.sigmoid(torch.tensor(gate_bias)).item():.4f} (={torch.sigmoid(torch.tensor(gate_bias)).item()*100:.1f}% of induced signal passes through)")
print()

# Check MIG gate initialization  
mig = MIGAttention(hidden_size=128, num_attention_heads=8, gate_dim=64, keep_ratio=0.7)
final_layer = mig.router[-1]
mig_bias = final_layer.bias.data.mean().item()
print(f"MIG gate init: bias={mig_bias:.4f}, sigmoid(bias)={torch.sigmoid(torch.tensor(mig_bias)).item():.4f}")
print(f"  => Initial gate output: ~{torch.sigmoid(torch.tensor(mig_bias)).item():.4f} ({torch.sigmoid(torch.tensor(mig_bias)).item()*100:.1f}% of signal passes to attention)")
print()

# Now check: at final trained MIG gate values from the aux_loss_series
# MIG aux_loss = mean(gate), and it's ~0.001172 (unscaled) -> 0.1172 mean gate
# Wait, aux_loss_series is SCALED by mig_lambda. Let me compute the unscaled value.
for f in sorted(glob.glob("reports/*.json")):
    if f.endswith("manifest.json") or "fair_bench" in f:
        continue
    with open(f) as fh:
        d = json.load(fh)
    cfg = d.get("config", {})
    if cfg.get("device") != "cuda":
        continue
    arch = cfg.get("arch")
    ds = cfg.get("dataset")
    m = d["metrics"]
    
    if arch == "mig":
        aux_mean = m.get("mig_aux_loss_mean", 0)
        mig_lam = cfg.get("mig_lambda", 0.01)
        # aux_loss_series stores scaled values (aux * lambda)
        # mig_aux_loss_mean is the mean of scaled values
        unscaled = aux_mean / mig_lam if mig_lam > 0 else 0
        print(f"MIG/{ds}: mig_aux_loss_mean(scaled)={aux_mean:.6f}, mig_lambda={mig_lam}")
        print(f"  Unscaled mean gate value = {unscaled:.4f} (= mean sigmoid output)")
        print(f"  => Gates are ~{unscaled*100:.1f}% open on average")
        
    elif arch == "sil":
        sil_aux_mean = m.get("sil_aux_loss_mean", 0)
        sil_lam = cfg.get("sil_lambda", 0.01)
        # sil_aux = -mean_entropy, scaled by sil_lambda
        unscaled = sil_aux_mean / sil_lam if sil_lam > 0 else 0
        entropy = -unscaled  # positive entropy
        import math
        max_ent = math.log(32)  # ln(num_rules)
        print(f"SIL/{ds}: sil_aux_loss_mean(scaled)={sil_aux_mean:.6f}, sil_lambda={sil_lam}")
        print(f"  Mean entropy = {entropy:.4f} (max possible = ln(32) = {max_ent:.4f})")
        print(f"  Rule diversity = {entropy/max_ent*100:.1f}% of maximum")
        print()
