import json, math, glob
# MIG: aux_loss = mean(sigmoid(gate)), scaled by lambda
# So aux_loss_scaled / lambda = mean gate activation
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
        lam = cfg.get("mig_lambda", 0.01)
        unscaled = aux_mean / lam if lam > 0 else 0
        print(f"MIG/{ds}: aux_mean_scaled={aux_mean:.6f}, lambda={lam}, mean_gate={unscaled:.4f} ({unscaled*100:.1f}% open)")
    elif arch == "sil":
        aux_mean = m.get("sil_aux_loss_mean", 0)
        lam = cfg.get("sil_lambda", 0.01)
        unscaled = aux_mean / lam if lam > 0 else 0
        entropy = -unscaled
        max_ent = math.log(cfg.get("sil_num_latent_rules", 32))
        print(f"SIL/{ds}: aux_mean_scaled={aux_mean:.6f}, lambda={lam}, mean_entropy={entropy:.4f} (max={max_ent:.4f}, diversity={entropy/max_ent*100:.1f}%)")
