"""Deep analysis of CUDA benchmark results for MIG/SIL evaluation."""
import json
import pathlib
import sys

reports_dir = pathlib.Path("reports")
rows = []
full_data = {}  # arch -> dataset -> full report data

for p in sorted(reports_dir.glob("2026*.json")):
    r = json.loads(p.read_text())
    cfg = r.get("config", {})
    met = r.get("metrics", {})
    model = r.get("model", {})
    if cfg.get("device") != "cuda":
        continue
    loss_series = met.get("loss_series", [])
    lr_series = met.get("lr_series", [])
    first_loss = loss_series[0] if loss_series else None
    last_loss = loss_series[-1] if loss_series else None
    min_loss = min(loss_series) if loss_series else None
    # Loss at key checkpoints
    l50 = loss_series[49] if len(loss_series) > 49 else None
    l100 = loss_series[99] if len(loss_series) > 99 else None
    l200 = loss_series[199] if len(loss_series) > 199 else None
    l300 = loss_series[299] if len(loss_series) > 299 else None
    l500 = loss_series[499] if len(loss_series) > 499 else None

    arch = cfg.get("arch")
    dataset = cfg.get("dataset")
    row = {
        "arch": arch,
        "dataset": dataset,
        "params": model.get("total_parameters"),
        "embed_params": model.get("embedding_parameters"),
        "mixin_params": model.get("mixin_parameters"),
        "ffn_params": model.get("ffn_parameters"),
        "tok_s": met.get("tokens_per_s"),
        "step_ms": met.get("train_step_ms_mean"),
        "fwd_ms": met.get("forward_ms_mean"),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "min_loss": min_loss,
        "l50": l50,
        "l100": l100,
        "l200": l200,
        "l300": l300,
        "l500": l500,
        "peak_mb": met.get("peak_mem_mb"),
        "steps": cfg.get("steps"),
        "loss_series": loss_series,
    }
    rows.append(row)
    full_data.setdefault(arch, {})[dataset] = row

rows.sort(key=lambda r: (r["dataset"], r["arch"]))

# ═══════════════════════════════════════════════════════════════════
# TABLE 1: Full comparison
# ═══════════════════════════════════════════════════════════════════
print("=" * 110)
print("TABLE 1: CUDA 500-Step Benchmark Results (All Architectures)")
print("=" * 110)
hdr = f"{'Arch':<8} {'Dataset':<12} {'Params':>8} {'tok/s':>7} {'step_ms':>8} {'L@1':>7} {'L@50':>7} {'L@100':>7} {'L@200':>7} {'L@500':>7} {'Min':>7} {'PeakMB':>7}"
print(hdr)
print("-" * 110)
for r in rows:
    l1 = f"{r['first_loss']:.4f}" if r["first_loss"] is not None else "   N/A"
    l50 = f"{r['l50']:.4f}" if r["l50"] is not None else "   N/A"
    l100 = f"{r['l100']:.4f}" if r["l100"] is not None else "   N/A"
    l200 = f"{r['l200']:.4f}" if r["l200"] is not None else "   N/A"
    l500 = f"{r['l500']:.4f}" if r["l500"] is not None else "   N/A"
    mn = f"{r['min_loss']:.4f}" if r["min_loss"] is not None else "   N/A"
    print(
        f"{r['arch']:<8} {r['dataset']:<12} {r['params']:>8d} {r['tok_s']:>7.0f} {r['step_ms']:>8.1f} "
        f"{l1:>7} {l50:>7} {l100:>7} {l200:>7} {l500:>7} {mn:>7} {r['peak_mb']:>7.0f}"
    )

# ═══════════════════════════════════════════════════════════════════
# TABLE 2: Ranking by final loss per dataset
# ═══════════════════════════════════════════════════════════════════
for ds in ["synthetic", "tinystories"]:
    subset = [r for r in rows if r["dataset"] == ds]
    subset.sort(key=lambda r: r["last_loss"] if r["last_loss"] is not None else 999)
    print(f"\n{'=' * 70}")
    print(f"RANKING: {ds.upper()} — Final Loss (lower is better)")
    print(f"{'=' * 70}")
    gqa_loss = None
    for i, r in enumerate(subset):
        if r["arch"] == "gqa":
            gqa_loss = r["last_loss"]
    for i, r in enumerate(subset):
        delta = ""
        if gqa_loss is not None and r["last_loss"] is not None and r["arch"] != "gqa":
            d = r["last_loss"] - gqa_loss
            pct = (d / gqa_loss) * 100 if gqa_loss != 0 else 0
            sign = "+" if d > 0 else ""
            delta = f"  (vs GQA: {sign}{pct:.1f}%)"
        star = " ★" if r["arch"] in ("mig", "sil") else ""
        print(f"  {i+1}. {r['arch']:<8} loss={r['last_loss']:.4f}  ({r['params']:,} params, {r['tok_s']:.0f} tok/s){delta}{star}")

# ═══════════════════════════════════════════════════════════════════
# TABLE 3: Parameter breakdown
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("PARAMETER BREAKDOWN (Mixin = attention/mixing layer params)")
print(f"{'=' * 80}")
print(f"{'Arch':<8} {'Total':>10} {'Embed':>10} {'Mixin':>10} {'FFN':>10} {'Mixin%':>7}")
print("-" * 55)
seen = set()
for r in rows:
    if r["arch"] in seen:
        continue
    seen.add(r["arch"])
    total = r["params"] or 1
    mixin = r["mixin_params"] or 0
    pct = (mixin / total) * 100
    print(
        f"{r['arch']:<8} {r['params']:>10,} {r['embed_params']:>10,} "
        f"{r['mixin_params']:>10,} {r['ffn_params']:>10,} {pct:>6.1f}%"
    )

# ═══════════════════════════════════════════════════════════════════
# TABLE 4: Convergence speed (steps to reach loss thresholds)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print("CONVERGENCE SPEED: Steps to reach loss thresholds (synthetic dataset)")
print(f"{'=' * 90}")
thresholds = [6.0, 4.0, 2.0, 1.0, 0.5]
print(f"{'Arch':<8}", end="")
for t in thresholds:
    print(f" {'L<'+str(t):>8}", end="")
print()
print("-" * 55)

for arch in sorted(full_data.keys()):
    if "synthetic" not in full_data[arch]:
        continue
    series = full_data[arch]["synthetic"]["loss_series"]
    print(f"{arch:<8}", end="")
    for t in thresholds:
        step = None
        for i, v in enumerate(series):
            if v < t:
                step = i + 1
                break
        if step is not None:
            print(f" {step:>8d}", end="")
        else:
            print(f" {'never':>8}", end="")
    print()

# ═══════════════════════════════════════════════════════════════════
# TABLE 5: Efficiency (loss per parameter, throughput-adjusted)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("EFFICIENCY: Final Loss per Million Parameters (synthetic)")
print(f"{'=' * 80}")
syn = [r for r in rows if r["dataset"] == "synthetic"]
syn.sort(key=lambda r: r["last_loss"] / (r["params"] / 1e6) if r["params"] else 999)
print(f"{'Arch':<8} {'FinalLoss':>10} {'Params(M)':>10} {'Loss/MParam':>12} {'tok/s/MParam':>13}")
print("-" * 55)
for r in syn:
    mp = r["params"] / 1e6
    lpm = r["last_loss"] / mp if mp > 0 else 0
    tsm = r["tok_s"] / mp if mp > 0 else 0
    print(f"{r['arch']:<8} {r['last_loss']:>10.4f} {mp:>10.3f} {lpm:>12.4f} {tsm:>13.1f}")

# ═══════════════════════════════════════════════════════════════════
# ANALYSIS: Synthetic vs TinyStories gap
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("SYNTHETIC VS TINYSTORIES GAP (measures generalization difficulty)")
print(f"{'=' * 80}")
print(f"{'Arch':<8} {'Synth Loss':>11} {'TS Loss':>11} {'Gap':>8} {'Gap%':>8}")
print("-" * 50)
for arch in sorted(full_data.keys()):
    syn_loss = full_data[arch].get("synthetic", {}).get("last_loss")
    ts_loss = full_data[arch].get("tinystories", {}).get("last_loss")
    if syn_loss is not None and ts_loss is not None:
        gap = ts_loss - syn_loss
        gap_pct = (gap / syn_loss) * 100 if syn_loss != 0 else 0
        print(f"{arch:<8} {syn_loss:>11.4f} {ts_loss:>11.4f} {gap:>8.4f} {gap_pct:>7.1f}%")
