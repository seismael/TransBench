"""Analyze aux loss trajectories for MIG, SIL, ASR, and GQA from CUDA reports."""
import json, glob, os

for f in sorted(glob.glob("reports/*.json")):
    if f.endswith("manifest.json") or "fair_bench" in f:
        continue
    with open(f) as fh:
        d = json.load(fh)
    dev = d.get("system_info", {}).get("device", "")
    arch = d.get("config", {}).get("arch", "")
    ds = d.get("config", {}).get("dataset", "")
    if dev == "cuda" and arch in ("mig", "sil", "gqa", "asr"):
        ls = d.get("loss_series", [])
        aux = d.get("aux_loss_series", [])
        sil_aux = d.get("sil_aux_loss_series", [])
        asr_aux = d.get("asr_aux_loss_series", [])
        total = d.get("total_loss_series", [])
        print(f"{os.path.basename(f)}: {arch}/{ds} device={dev}")
        print(f"  CE loss: len={len(ls)}")
        print(f"    step1={ls[0]:.4f}  step50={ls[49]:.4f}  step100={ls[99]:.4f}  step250={ls[249]:.4f}  step500={ls[-1]:.4f}")
        if aux:
            print(f"  MIG aux(scaled): len={len(aux)}")
            print(f"    step1={aux[0]:.6f}  step50={aux[49]:.6f}  step100={aux[99]:.6f}  step250={aux[249]:.6f}  step500={aux[-1]:.6f}")
        if sil_aux:
            print(f"  SIL aux(scaled): len={len(sil_aux)}")
            print(f"    step1={sil_aux[0]:.6f}  step50={sil_aux[49]:.6f}  step100={sil_aux[99]:.6f}  step250={sil_aux[249]:.6f}  step500={sil_aux[-1]:.6f}")
        if asr_aux:
            print(f"  ASR aux(scaled): len={len(asr_aux)}")
            print(f"    step1={asr_aux[0]:.6f}  step50={asr_aux[49]:.6f}  step100={asr_aux[99]:.6f}  step250={asr_aux[249]:.6f}  step500={asr_aux[-1]:.6f}")
        if total:
            print(f"  Total loss(CE+aux): len={len(total)}")
            print(f"    step1={total[0]:.4f}  step50={total[49]:.4f}  step100={total[99]:.4f}  step250={total[249]:.4f}  step500={total[-1]:.4f}")
        print()

# Now compare convergence curves between GQA (baseline) and MIG/SIL
print("="*80)
print("CONVERGENCE COMPARISON: GQA vs MIG vs SIL vs ASR (CUDA, synthetic)")
print("="*80)

archs_data = {}
for f in sorted(glob.glob("reports/*.json")):
    if f.endswith("manifest.json") or "fair_bench" in f:
        continue
    with open(f) as fh:
        d = json.load(fh)
    dev = d.get("system_info", {}).get("device", "")
    arch = d.get("config", {}).get("arch", "")
    ds = d.get("config", {}).get("dataset", "")
    if dev == "cuda" and ds == "synthetic":
        archs_data[arch] = d.get("loss_series", [])

checkpoints = [1, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
print(f"\n{'Step':>6}", end="")
for a in ["gqa", "mig", "sil", "asr", "mhla", "retnet", "mamba2", "rwkv6"]:
    print(f"  {a:>8}", end="")
print()
print("-" * 120)

for cp in checkpoints:
    print(f"{cp:>6}", end="")
    for a in ["gqa", "mig", "sil", "asr", "mhla", "retnet", "mamba2", "rwkv6"]:
        ls = archs_data.get(a, [])
        if cp <= len(ls):
            val = ls[cp - 1]
            print(f"  {val:>8.4f}", end="")
        else:
            print(f"  {'N/A':>8}", end="")
    print()

print("\n\n")
print("="*80)
print("CONVERGENCE COMPARISON: GQA vs MIG vs SIL vs ASR (CUDA, tinystories)")
print("="*80)

archs_data_ts = {}
for f in sorted(glob.glob("reports/*.json")):
    if f.endswith("manifest.json") or "fair_bench" in f:
        continue
    with open(f) as fh:
        d = json.load(fh)
    dev = d.get("system_info", {}).get("device", "")
    arch = d.get("config", {}).get("arch", "")
    ds = d.get("config", {}).get("dataset", "")
    if dev == "cuda" and ds == "tinystories":
        archs_data_ts[arch] = d.get("loss_series", [])

print(f"\n{'Step':>6}", end="")
for a in ["gqa", "mig", "sil", "asr", "mhla", "retnet", "mamba2", "rwkv6"]:
    print(f"  {a:>8}", end="")
print()
print("-" * 120)

for cp in checkpoints:
    print(f"{cp:>6}", end="")
    for a in ["gqa", "mig", "sil", "asr", "mhla", "retnet", "mamba2", "rwkv6"]:
        ls = archs_data_ts.get(a, [])
        if cp <= len(ls):
            val = ls[cp - 1]
            print(f"  {val:>8.4f}", end="")
        else:
            print(f"  {'N/A':>8}", end="")
    print()

# Relative performance vs GQA at each checkpoint
print("\n\n")
print("="*80)
print("RELATIVE PERFORMANCE vs GQA (negative = better than GQA)")
print("="*80)

for ds_name, data in [("synthetic", archs_data), ("tinystories", archs_data_ts)]:
    print(f"\n--- {ds_name.upper()} ---")
    gqa_ls = data.get("gqa", [])
    if not gqa_ls:
        print("  No GQA data!")
        continue
    print(f"{'Step':>6}", end="")
    for a in ["mig", "sil", "asr"]:
        print(f"  {a:>10}", end="")
    print()
    for cp in checkpoints:
        if cp > len(gqa_ls):
            break
        gqa_val = gqa_ls[cp - 1]
        print(f"{cp:>6}", end="")
        for a in ["mig", "sil", "asr"]:
            ls = data.get(a, [])
            if cp <= len(ls):
                val = ls[cp - 1]
                pct = ((val - gqa_val) / gqa_val) * 100
                print(f"  {pct:>+9.2f}%", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()
