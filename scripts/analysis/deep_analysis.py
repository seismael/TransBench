"""Deep analysis of CUDA benchmark results - correct field paths."""
import json, glob, os

def load_cuda_reports():
    """Load all CUDA reports into a dict keyed by (arch, dataset)."""
    reports = {}
    for f in sorted(glob.glob("reports/*.json")):
        if f.endswith("manifest.json") or "fair_bench" in f:
            continue
        with open(f) as fh:
            d = json.load(fh)
        cfg = d.get("config", {})
        dev = cfg.get("device", "")
        if dev != "cuda":
            continue
        arch = cfg.get("arch", "")
        ds = cfg.get("dataset", "")
        reports[(arch, ds)] = d
    return reports

reports = load_cuda_reports()
print(f"Loaded {len(reports)} CUDA reports")
for key in sorted(reports.keys()):
    print(f"  {key[0]:>8} / {key[1]}")

ARCHS_ORDER = ["gqa", "mig", "sil", "asr", "mhla", "retnet", "mamba2", "rwkv6"]
WARMUP = 10  # first 10 entries are warmup

# ============================================================
# TABLE 1: Aux loss details for MIG, SIL, ASR
# ============================================================
print("\n" + "="*80)
print("TABLE 1: AUX LOSS DETAILS (MIG, SIL, ASR)")
print("="*80)
for arch in ["mig", "sil", "asr"]:
    for ds in ["synthetic", "tinystories"]:
        d = reports.get((arch, ds))
        if not d:
            continue
        m = d["metrics"]
        cfg = d["config"]
        ls = m["loss_series"][WARMUP:]  # skip warmup
        total = m.get("total_loss_series", [])[WARMUP:]
        
        print(f"\n--- {arch.upper()} / {ds} ---")
        print(f"  CE loss: first={ls[0]:.4f}  step50={ls[49]:.4f}  step100={ls[99]:.4f}  step250={ls[249]:.4f}  final={ls[-1]:.4f}")
        if total:
            print(f"  Total(CE+aux): first={total[0]:.4f}  step50={total[49]:.4f}  step100={total[99]:.4f}  step250={total[249]:.4f}  final={total[-1]:.4f}")
            # Compute aux contribution at each checkpoint
            for cp_name, idx in [("step1", 0), ("step50", 49), ("step100", 99), ("step250", 249), ("final", -1)]:
                diff = total[idx] - ls[idx]
                pct = (diff / total[idx]) * 100 if total[idx] != 0 else 0
                pass
            aux_diffs = [(total[i] - ls[i]) for i in range(len(ls))]
            print(f"  Aux contribution: first={aux_diffs[0]:.6f}  step50={aux_diffs[49]:.6f}  step100={aux_diffs[99]:.6f}  step250={aux_diffs[249]:.6f}  final={aux_diffs[-1]:.6f}")
            print(f"  Aux % of total: first={aux_diffs[0]/total[0]*100:.2f}%  final={aux_diffs[-1]/total[-1]*100:.3f}%")
        
        print(f"  Config: mig_lambda={cfg.get('mig_lambda',0)}, sil_lambda={cfg.get('sil_lambda',0)}, asr_lambda={cfg.get('asr_lambda',0)}")

# ============================================================
# TABLE 2: Full convergence curves (synthetic)
# ============================================================
print("\n\n" + "="*100)
print("TABLE 2: CONVERGENCE CURVES - SYNTHETIC (CE loss, warmup excluded)")
print("="*100)

checkpoints = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
print(f"{'Step':>6}", end="")
for a in ARCHS_ORDER:
    print(f"  {a:>8}", end="")
print()
print("-" * 106)

for cp in checkpoints:
    print(f"{cp:>6}", end="")
    for a in ARCHS_ORDER:
        d = reports.get((a, "synthetic"))
        if d:
            ls = d["metrics"]["loss_series"][WARMUP:]
            if cp <= len(ls):
                print(f"  {ls[cp-1]:>8.4f}", end="")
            else:
                print(f"  {'N/A':>8}", end="")
        else:
            print(f"  {'---':>8}", end="")
    print()

# ============================================================
# TABLE 3: Full convergence curves (tinystories)
# ============================================================
print("\n\n" + "="*100)
print("TABLE 3: CONVERGENCE CURVES - TINYSTORIES (CE loss, warmup excluded)")
print("="*100)

print(f"{'Step':>6}", end="")
for a in ARCHS_ORDER:
    print(f"  {a:>8}", end="")
print()
print("-" * 106)

for cp in checkpoints:
    print(f"{cp:>6}", end="")
    for a in ARCHS_ORDER:
        d = reports.get((a, "tinystories"))
        if d:
            ls = d["metrics"]["loss_series"][WARMUP:]
            if cp <= len(ls):
                print(f"  {ls[cp-1]:>8.4f}", end="")
            else:
                print(f"  {'N/A':>8}", end="")
        else:
            print(f"  {'---':>8}", end="")
    print()

# ============================================================
# TABLE 4: Relative % vs GQA at each checkpoint
# ============================================================
print("\n\n" + "="*100)
print("TABLE 4: RELATIVE PERFORMANCE vs GQA (% difference, negative = BETTER than GQA)")
print("="*100)

for ds in ["synthetic", "tinystories"]:
    print(f"\n--- {ds.upper()} ---")
    gqa_d = reports.get(("gqa", ds))
    if not gqa_d:
        print("  No GQA data!")
        continue
    gqa_ls = gqa_d["metrics"]["loss_series"][WARMUP:]
    
    print(f"{'Step':>6}", end="")
    for a in ["mig", "sil", "asr", "mhla", "retnet", "mamba2", "rwkv6"]:
        print(f"  {a:>8}", end="")
    print()
    print("-" * 88)
    
    for cp in checkpoints:
        if cp > len(gqa_ls):
            break
        gqa_val = gqa_ls[cp - 1]
        print(f"{cp:>6}", end="")
        for a in ["mig", "sil", "asr", "mhla", "retnet", "mamba2", "rwkv6"]:
            d = reports.get((a, ds))
            if d:
                ls = d["metrics"]["loss_series"][WARMUP:]
                if cp <= len(ls):
                    val = ls[cp - 1]
                    pct = ((val - gqa_val) / gqa_val) * 100
                    print(f"  {pct:>+7.1f}%", end="")
                else:
                    print(f"  {'N/A':>8}", end="")
            else:
                print(f"  {'---':>8}", end="")
        print()

# ============================================================
# TABLE 5: Rankings at each checkpoint
# ============================================================
print("\n\n" + "="*100)
print("TABLE 5: RANKINGS AT KEY CHECKPOINTS")
print("="*100)

for ds in ["synthetic", "tinystories"]:
    print(f"\n--- {ds.upper()} ---")
    for cp in [1, 25, 50, 100, 200, 500]:
        results = []
        for a in ARCHS_ORDER:
            d = reports.get((a, ds))
            if d:
                ls = d["metrics"]["loss_series"][WARMUP:]
                if cp <= len(ls):
                    results.append((a, ls[cp-1]))
        results.sort(key=lambda x: x[1])
        ranking_str = "  ".join(f"{r+1}.{a}({v:.4f})" for r, (a, v) in enumerate(results))
        print(f"  Step {cp:>3}: {ranking_str}")

# ============================================================
# TABLE 6: Parameter efficiency
# ============================================================
print("\n\n" + "="*100)
print("TABLE 6: PARAMETER EFFICIENCY (synthetic final loss / total params)")
print("="*100)

print(f"{'Arch':>8} {'Params':>10} {'Mixin':>10} {'Mixin%':>8} {'SynthLoss':>10} {'TSLoss':>10} {'Loss/MParam(S)':>15} {'Loss/MParam(T)':>15}")
print("-" * 100)

for a in ARCHS_ORDER:
    d_s = reports.get((a, "synthetic"))
    d_t = reports.get((a, "tinystories"))
    if d_s:
        model = d_s.get("model", {})
        params = model.get("total_parameters", 0)
        mixin = model.get("mixin_parameters", 0)
        mixin_pct = (mixin / params * 100) if params > 0 else 0
        ls_s = d_s["metrics"]["loss_series"][WARMUP:]
        final_s = ls_s[-1] if ls_s else float("nan")
        
        ls_t = d_t["metrics"]["loss_series"][WARMUP:] if d_t else []
        final_t = ls_t[-1] if ls_t else float("nan")
        
        eff_s = final_s / (params / 1e6) if params > 0 else float("nan")
        eff_t = final_t / (params / 1e6) if params > 0 else float("nan")
        
        print(f"{a:>8} {params:>10,} {mixin:>10,} {mixin_pct:>7.1f}% {final_s:>10.4f} {final_t:>10.4f} {eff_s:>15.4f} {eff_t:>15.4f}")

# ============================================================
# TABLE 7: Speed metrics
# ============================================================
print("\n\n" + "="*100)
print("TABLE 7: SPEED METRICS (CUDA)")
print("="*100)

print(f"{'Arch':>8} {'Dataset':>12} {'Fwd ms':>8} {'Train ms':>10} {'Tok/s':>10} {'Loss':>10}")
print("-" * 70)

for a in ARCHS_ORDER:
    for ds in ["synthetic", "tinystories"]:
        d = reports.get((a, ds))
        if d:
            m = d["metrics"]
            print(f"{a:>8} {ds:>12} {m['forward_ms_mean']:>8.2f} {m['train_step_ms_mean']:>10.2f} {m['tokens_per_s']:>10.0f} {m['loss_series'][WARMUP:][-1]:>10.4f}")

# ============================================================
# TABLE 8: SIL deep-dive - Gate behavior
# ============================================================
print("\n\n" + "="*100)
print("TABLE 8: SIL ANALYSIS - Why is it underperforming?")
print("="*100)

for ds in ["synthetic", "tinystories"]:
    sil_d = reports.get(("sil", ds))
    gqa_d = reports.get(("gqa", ds))
    if not sil_d or not gqa_d:
        continue
    sil_ls = sil_d["metrics"]["loss_series"][WARMUP:]
    gqa_ls = gqa_d["metrics"]["loss_series"][WARMUP:]
    sil_total = sil_d["metrics"].get("total_loss_series", [])[WARMUP:]
    
    print(f"\n--- {ds.upper()} ---")
    print(f"  SIL config: num_latent_rules={sil_d['config'].get('sil_num_latent_rules')}, "
          f"temperature={sil_d['config'].get('sil_temperature')}, "
          f"hard_train={sil_d['config'].get('sil_hard_train')}, "
          f"sil_lambda={sil_d['config'].get('sil_lambda')}")
    
    # Check how SIL CE compares to GQA at early, mid, late
    for cp_name, idx in [("Step 1", 0), ("Step 25", 24), ("Step 50", 49), 
                          ("Step 100", 99), ("Step 200", 199), ("Step 500", 499)]:
        sil_v = sil_ls[idx]
        gqa_v = gqa_ls[idx]
        gap_pct = ((sil_v - gqa_v) / gqa_v) * 100
        aux_contrib = (sil_total[idx] - sil_ls[idx]) if sil_total else 0
        print(f"  {cp_name:>10}: SIL={sil_v:.4f}  GQA={gqa_v:.4f}  gap={gap_pct:+.1f}%  aux_loss={aux_contrib:.6f}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n\n" + "="*100)
print("SUMMARY: FINAL LOSS RANKINGS + ANALYSIS")
print("="*100)

for ds in ["synthetic", "tinystories"]:
    print(f"\n--- {ds.upper()} ---")
    results = []
    for a in ARCHS_ORDER:
        d = reports.get((a, ds))
        if d:
            ls = d["metrics"]["loss_series"][WARMUP:]
            results.append((a, ls[-1]))
    results.sort(key=lambda x: x[1])
    
    gqa_loss = dict(results).get("gqa", float("inf"))
    for rank, (a, loss) in enumerate(results, 1):
        vs_gqa = ((loss - gqa_loss) / gqa_loss) * 100
        status = "BETTER" if vs_gqa < -1 else ("SAME" if abs(vs_gqa) <= 1 else "WORSE")
        print(f"  #{rank:>2} {a:>8}: {loss:.4f}  ({vs_gqa:+.1f}% vs GQA) [{status}]")
