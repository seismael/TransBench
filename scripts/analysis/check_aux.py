"""Check MIG/SIL auxiliary loss values and hyperparameters."""
import json, pathlib

for p in sorted(pathlib.Path("reports").glob("2026*.json")):
    r = json.loads(p.read_text())
    cfg = r["config"]
    if cfg.get("device") != "cuda":
        continue
    arch = cfg["arch"]
    if arch in ("mig", "sil"):
        met = r["metrics"]
        aux = met.get("aux_loss_series", [])
        sil_aux = met.get("sil_aux_loss_series", [])
        total = met.get("total_loss_series", [])
        ds = cfg["dataset"]
        print(f"{arch} {ds}: mig_lambda={cfg.get('mig_lambda')}, sil_lambda={cfg.get('sil_lambda')}, "
              f"mig_kr={cfg.get('mig_keep_ratio')}, gate_dim={cfg.get('mig_gate_dim')}, "
              f"sil_nr={cfg.get('sil_num_latent_rules')}, sil_temp={cfg.get('sil_temperature')}")
        if aux:
            print(f"  aux_loss: first={aux[0]:.6f} last={aux[-1]:.6f}")
        if sil_aux:
            print(f"  sil_aux:  first={sil_aux[0]:.6f} last={sil_aux[-1]:.6f}")
        if total:
            print(f"  total:    first={total[0]:.4f} last={total[-1]:.4f}")
        else:
            print(f"  NO total_loss_series -> aux losses may not be active!")
        
        # Check all metric keys
        print(f"  metric_keys: {[k for k in met.keys() if 'aux' in k or 'loss' in k.lower() or 'total' in k]}")
        print()
