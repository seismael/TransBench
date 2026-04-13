"""Compare only REAL architectures (no FLA fallback stubs)."""
import json, os

reports = {
    "GQA":  "20260413_180253_gqa_abc677c1da.json",
    "MHLA": "20260413_181500_mhla_a399ece840.json",
    "ASR":  "20260413_192433_asr_437e9969e0.json",
    "MIG":  "20260413_184750_mig_2a0953165a.json",
    "SIL":  "20260413_190055_sil_a22eb4e909.json",
}

rdir = os.path.join(os.path.dirname(__file__), "..", "reports")
rows = []
for arch, fname in reports.items():
    with open(os.path.join(rdir, fname)) as f:
        r = json.load(f)
    m = r["metrics"]
    cfg = r.get("config", {})
    ls = m.get("loss_series", [])
    final = ls[-1] if ls else 0
    init = ls[0] if ls else 0
    fwd = m.get("forward_ms_mean", 0)
    step = m.get("train_step_ms_mean", 0)
    p50 = m.get("forward_ms_p50", 0)
    p95 = m.get("forward_ms_p95", 0)
    params = m.get("param_count", 0)
    gate_init = m.get("gate_mean_init", None)
    gate_final = m.get("gate_mean_final", None)
    rows.append(dict(
        arch=arch, loss=final, loss_init=init, fwd=fwd, step=step,
        p50=p50, p95=p95, params=params, gate_init=gate_init, gate_final=gate_final,
    ))

rows.sort(key=lambda x: x["loss"])
best = rows[0]["loss"]

print("=" * 78)
print("  FAIR COMPARISON — REAL ARCHITECTURES ONLY (2000 steps, CUDA sm_61)")
print("  Excludes: Mamba2, RetNet, RWKV6 (all ran trivial Conv1d fallbacks)")
print("=" * 78)
print()

# Loss table
hdr = f"{'Rank':<5} {'Arch':<6} {'Loss':>8} {'Delta':>8} {'Type':<10} {'Params':>9}"
print(hdr)
print("-" * len(hdr))
for i, r in enumerate(rows, 1):
    role = "BASELINE" if r["arch"] in ("GQA", "MHLA") else "NOVEL"
    delta = r["loss"] - best
    d = f"+{delta:.5f}" if delta > 0 else "  best"
    p = f"{r['params']:,}" if r["params"] else "?"
    print(f"{i:<5} {r['arch']:<6} {r['loss']:>8.5f} {d:>8} {role:<10} {p:>9}")

print()

# Speed table
print("  LATENCY")
hdr2 = f"{'Arch':<6} {'Fwd mean':>10} {'Fwd P50':>10} {'Fwd P95':>10} {'Step mean':>11}"
print(hdr2)
print("-" * len(hdr2))
for r in rows:
    print(f"{r['arch']:<6} {r['fwd']:>9.2f}ms {r['p50']:>9.2f}ms {r['p95']:>9.2f}ms {r['step']:>10.2f}ms")

print()

# Gate telemetry for MIG/SIL
gated = [r for r in rows if r["gate_init"] is not None]
if gated:
    print("  GATE TELEMETRY")
    hdr3 = f"{'Arch':<6} {'Gate init':>10} {'Gate final':>11} {'Movement':>10}"
    print(hdr3)
    print("-" * len(hdr3))
    for r in gated:
        gi = r["gate_init"]
        gf = r["gate_final"]
        mv = gf - gi if gi is not None and gf is not None else 0
        sign = "+" if mv >= 0 else ""
        print(f"{r['arch']:<6} {gi:>10.4f} {gf:>11.4f} {sign}{mv:>9.4f}")
    print()

# Convergence (loss drop)
print("  CONVERGENCE (loss init -> final)")
hdr4 = f"{'Arch':<6} {'Init':>8} {'Final':>8} {'Drop':>8} {'Drop%':>7}"
print(hdr4)
print("-" * len(hdr4))
for r in rows:
    drop = r["loss_init"] - r["loss"]
    pct = (drop / r["loss_init"] * 100) if r["loss_init"] else 0
    print(f"{r['arch']:<6} {r['loss_init']:>8.5f} {r['loss']:>8.5f} {drop:>8.5f} {pct:>6.1f}%")

print()
print("  VERDICT")
print("  -------")
baseline_best = min(r["loss"] for r in rows if r["arch"] in ("GQA", "MHLA"))
novel_best = min(r["loss"] for r in rows if r["arch"] not in ("GQA", "MHLA"))
gap = novel_best - baseline_best
print(f"  Best baseline: {baseline_best:.5f} (MHLA)")
print(f"  Best novel:    {novel_best:.5f} (ASR)")
print(f"  Gap:           +{gap:.5f} — novel architectures {'TRAIL' if gap > 0 else 'LEAD'} baselines")
