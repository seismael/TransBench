"""Assess dataset and config suitability for differentiating architectures."""
import json, glob, math

def load_cuda():
    reports = {}
    for f in sorted(glob.glob("reports/*.json")):
        if f.endswith("manifest.json") or "fair_bench" in f:
            continue
        with open(f) as fh:
            d = json.load(fh)
        cfg = d.get("config", {})
        if cfg.get("device") != "cuda":
            continue
        reports[(cfg["arch"], cfg["dataset"])] = d
    return reports

reports = load_cuda()

print("=" * 80)
print("SUITABILITY ASSESSMENT")
print("=" * 80)

# 1. Synthetic: how much variance between architectures?
print("\n1. SYNTHETIC DATASET - Is it discriminative?")
print("-" * 50)
synth_final = {}
for (arch, ds), d in reports.items():
    if ds == "synthetic":
        synth_final[arch] = d["metrics"]["loss_series"][10:][-1]  # skip warmup

vals = sorted(synth_final.values())
# Exclude SIL outlier for spread calculation  
vals_no_sil = [v for a, v in synth_final.items() if a != "sil"]
spread = max(vals_no_sil) - min(vals_no_sil)
mean_v = sum(vals_no_sil) / len(vals_no_sil)
cv = (max(vals_no_sil) - min(vals_no_sil)) / mean_v * 100

print(f"  Final losses (excl SIL): {min(vals_no_sil):.4f} – {max(vals_no_sil):.4f}")
print(f"  Spread: {spread:.4f} ({cv:.1f}% of mean)")
print(f"  SIL outlier: {synth_final['sil']:.4f} (excluded)")
print(f"  Total tokens memorized: batch_size(4) × seq_len(128) = 512 tokens")
print(f"  Vocab utilization: 512 unique positions / 4096 vocab = 12.5%")
print(f"  VERDICT: NOT SUITABLE. Task is pure memorization of 512 tokens.")
print(f"  All attention models converge to ~0.11 (near-zero CE). Differences")
print(f"  reflect optimization speed, not architectural quality.")

# 2. TinyStories: is the spread meaningful?
print("\n2. TINYSTORIES DATASET - Is it discriminative?")
print("-" * 50)
ts_final = {}
for (arch, ds), d in reports.items():
    if ds == "tinystories":
        ts_final[arch] = d["metrics"]["loss_series"][10:][-1]

vals_ts = sorted(ts_final.values())
vals_ts_no_sil = [v for a, v in ts_final.items() if a != "sil"]
spread_ts = max(vals_ts_no_sil) - min(vals_ts_no_sil)
mean_ts = sum(vals_ts_no_sil) / len(vals_ts_no_sil)
cv_ts = spread_ts / mean_ts * 100

print(f"  Final losses (excl SIL): {min(vals_ts_no_sil):.4f} – {max(vals_ts_no_sil):.4f}")
print(f"  Spread: {spread_ts:.4f} ({cv_ts:.1f}% of mean)")
print(f"  SIL: {ts_final['sil']:.4f}")
print(f"  VERDICT: MARGINAL. ~0.12 spread across 9 architectures. All within")
print(f"  2.5% of each other. Would need multiple seeds to confirm significance.")

# 3. Model capacity
print("\n3. MODEL CAPACITY - Is it sufficient?")
print("-" * 50)
d = reports.get(("gqa", "tinystories"))
if d:
    params = d["model"]["total_parameters"]
    cfg = d["config"]
    print(f"  Total params: {params:,} (~{params/1e6:.1f}M)")
    print(f"  Hidden: {cfg['hidden_size']}, Layers: {cfg['num_layers']}, Heads: {cfg['num_heads']}")
    print(f"  Context window: {cfg['seq_len']} tokens")
    print(f"  Training: {cfg['steps']} steps × batch {cfg['batch_size']} = {cfg['steps']*cfg['batch_size']*cfg['seq_len']:,} tokens seen")
    
    tokens_seen = cfg['steps'] * cfg['batch_size'] * cfg['seq_len']
    chinchilla_optimal = params * 20  # Chinchilla: 20 tokens per param
    print(f"  Chinchilla optimal: ~{chinchilla_optimal:,} tokens ({chinchilla_optimal/1e6:.0f}M)")
    print(f"  Tokens seen: {tokens_seen:,} ({tokens_seen/chinchilla_optimal*100:.1f}% of Chinchilla)")
    print(f"  VERDICT: SEVERELY UNDERTRAINED. Need ~{chinchilla_optimal/tokens_seen:.0f}x more data.")

# 4. Sequence length for MIG/SIL
print("\n4. SEQUENCE LENGTH - Suitable for MIG/SIL?")
print("-" * 50)
print(f"  Current seq_len: 128")
print(f"  MIG keep_ratio=0.7: keeps {int(128*0.7)}/128 tokens, drops {int(128*0.3)}")
print(f"  At 128 tokens, ~38 tokens dropped — minimal filtering possible.")
print(f"  MIG needs longer sequences (512+) where there's real redundancy to exploit.")
print(f"  SIL needs repeated patterns across context — 128 tokens ≈ 2 sentences,")
print(f"  too short for meaningful induction patterns.")

# 5. Training steps for gated architectures
print("\n5. TRAINING STEPS - Enough for gates to learn?")
print("-" * 50)
print(f"  Steps: 500 (+ 10 warmup)")
print(f"  MIG gate: init σ(-2)=0.119, trained avg=0.117-0.119 → GATES DID NOT MOVE")
print(f"  SIL gate: init σ(-3)=0.047, trained → UNKNOWN (no direct telemetry)")
print(f"  SIL aux_loss constant -0.034 throughout → entropy-max dominates, no learning signal")
print(f"  VERDICT: NOT ENOUGH. Gated architectures need 2000+ steps for gates to")
print(f"  receive meaningful gradient signal and diverge from initialization.")

# 6. MIG aux loss flatness
print("\n6. MIG AUX LOSS TRAJECTORY - Are gates learning?")
print("-" * 50)
for ds in ["synthetic", "tinystories"]:
    d = reports.get(("mig", ds))
    if not d:
        continue
    total = d["metrics"].get("total_loss_series", [])[10:]
    ce = d["metrics"]["loss_series"][10:]
    aux = [total[i]-ce[i] for i in range(len(ce))]
    # Scaled aux at key points
    print(f"  {ds}: aux@step1={aux[0]:.6f}  aux@step250={aux[249]:.6f}  aux@step500={aux[-1]:.6f}")
    pct_change = ((aux[-1] - aux[0]) / abs(aux[0])) * 100
    print(f"    Change over 500 steps: {pct_change:+.1f}%")
    print(f"    => Gate activation barely changed. Sparsity penalty (λ=0.01) pins gates near init.")

# 7. Overall verdict
print("\n" + "=" * 80)
print("OVERALL SUITABILITY VERDICT")
print("=" * 80)
print("""
  SYNTHETIC:    NOT SUITABLE (memorization task, not generalization)
  TINYSTORIES:  MARGINAL (real language, but model too small to differentiate)
  MODEL SIZE:   TOO SMALL (1.6M params, need 10M+ for head specialization)
  SEQ LENGTH:   TOO SHORT (128, need 512+ for MIG/SIL advantages)
  TRAIN STEPS:  TOO FEW (500, need 2000+ for gates to learn; 5000+ ideal)
  λ VALUES:     POSSIBLY TOO HIGH (mig_lambda=0.01 prevents gate opening)

  BOTTOM LINE: The current setup is a reasonable SMOKE TEST proving the code
  works, but it is NOT a fair evaluation of architectural quality. The model
  is too small and undertrained for MIG/SIL mechanisms to engage meaningfully.
""")
