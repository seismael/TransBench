from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from transbench.benchmark import BenchmarkConfig, run_benchmark
from transbench.datasets import default_cache_dir, list_dataset_ids, prepare_tinystories
from transbench.reporting import ReportsConfig, RunMeta, write_report_with_meta
from transbench.suite import load_suite_toml, run_suite
from transbench.clean import apply_clean_plan, build_clean_plan


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency).

    Loads KEY=VALUE pairs into os.environ if not already set.
    """

    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            os.environ.setdefault(key, value)
    except Exception:
        return


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transbench")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="Run a micro-benchmark and emit a JSON report")
    bench.add_argument(
        "--arch",
        default="gqa",
        help="Architecture: gqa|mig|sil|asr|mhla|mamba2|rwkv6|retnet",
    )
    bench.add_argument("--all", action="store_true", help="Run all supported architectures")
    bench.add_argument("--num-layers", type=int, default=8)
    bench.add_argument("--hidden-size", type=int, default=512)
    bench.add_argument("--ffn-mult", type=float, default=4.0, help="FFN intermediate multiplier")
    bench.add_argument("--num-heads", type=int, default=8)
    bench.add_argument("--num-kv-heads", type=int, default=None)
    bench.add_argument("--vocab-size", type=int, default=32000)
    bench.add_argument("--initializer-range", type=float, default=0.02)
    bench.add_argument("--seq-len", type=int, default=256)
    bench.add_argument("--batch-size", type=int, default=2)
    bench.add_argument("--warmup", type=int, default=0)
    bench.add_argument("--steps", type=int, default=50)
    bench.add_argument("--lr", type=float, default=2e-4, help="Optimizer learning rate (default: 2e-4)")
    bench.add_argument(
        "--dataset",
        default="tinystories",
        choices=list_dataset_ids(),
        help="Dataset: tinystories|synthetic|zeros|ramp (default: tinystories)",
    )
    bench.add_argument(
        "--tokenizer-model",
        default="gpt2",
        help="Tokenizer model id for tinystories tokenization (default: gpt2)",
    )
    bench.add_argument("--min-lr", type=float, default=1e-6, help="Min LR for cosine schedule (default: 1e-6)")
    bench.add_argument(
        "--mig-gate-dim",
        type=int,
        default=64,
        help="MIG: gate MLP hidden dim (default: 64)",
    )
    bench.add_argument(
        "--mig-lambda",
        type=float,
        default=0.01,
        help="MIG: sparsity loss weight (default: 0.01)",
    )
    bench.add_argument(
        "--mig-keep-ratio",
        type=float,
        default=0.7,
        help="MIG: fraction of tokens to keep for top-k dropping (default: 0.7)",
    )
    bench.add_argument(
        "--mig-layer-keep-ratios",
        type=str,
        default=None,
        help="MIG A-MIG: per-layer keep ratios as comma-separated floats (e.g. 0.05,0.05,0.05,1,1,1,1,1)",
    )
    bench.add_argument(
        "--poison-ratio",
        type=float,
        default=0.85,
        help="Poisoned Needle: fraction of center tokens to corrupt (default: 0.85)",
    )

    bench.add_argument(
        "--sil-num-rules",
        type=int,
        default=64,
        help="SIL: number of latent rules (default: 64)",
    )
    bench.add_argument(
        "--sil-temperature",
        type=float,
        default=0.5,
        help="SIL: Gumbel-Softmax temperature (default: 0.5)",
    )
    bench.add_argument(
        "--sil-hard-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="SIL: use hard (one-hot) sampling during training (default: false)",
    )
    bench.add_argument(
        "--sil-hard-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SIL: use hard (one-hot) sampling during eval (default: true)",
    )

    bench.add_argument(
        "--asr-noise-std",
        type=float,
        default=0.3,
        help="ASR: input noise std for invariance regularization (default: 0.3)",
    )
    bench.add_argument(
        "--asr-lambda",
        type=float,
        default=0.1,
        help="ASR: auxiliary loss weight (default: 0.1)",
    )
    bench.add_argument(
        "--sil-lambda",
        type=float,
        default=0.01,
        help="SIL: entropy-maximization auxiliary loss weight (default: 0.01)",
    )
    bench.add_argument("--seed", type=int, default=None)
    bench.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    bench.add_argument("--dtype", default=None, help="float32|bfloat16|float16 (default: auto)")
    bench.add_argument("--reports-dir", default="reports")
    bench.add_argument(
        "--cache-dir",
        default=str(default_cache_dir()),
        help="Cache dir for datasets/tokenizers (default: ~/.cache/transbench/datasets)",
    )
    bench.add_argument(
        "--offline",
        action="store_true",
        help="Reuse cached dataset/tokenizer only (no network)",
    )
    bench.add_argument("--run-name", default=None)
    bench.add_argument("--tag", action="append", default=[])
    bench.add_argument("--notes", default=None)
    bench.add_argument("--raw", action="store_true", help="Include raw timing arrays in the report")
    bench.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite stable report filename derived from config",
    )

    suite = sub.add_parser("suite", help="Run a suite of benchmarks from a TOML file")
    suite.add_argument("--config", required=True)
    suite.add_argument("--reports-dir", default="reports")

    manifest = sub.add_parser("make-manifest", help="Rebuild reports/manifest.json")
    manifest.add_argument("--reports-dir", default="reports")

    serve = sub.add_parser("serve-dashboard", help="Serve dashboard + reports via http.server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    clean = sub.add_parser("clean", help="Remove generated reports/caches (safe housekeeping)")
    clean.add_argument("--reports", action="store_true")
    clean.add_argument("--caches", action="store_true")
    clean.add_argument("--runs", action="store_true")
    clean.add_argument("--all", action="store_true")
    clean.add_argument("--dry-run", action="store_true")
    clean.add_argument("--yes", action="store_true")

    prep = sub.add_parser("prepare", help="Download/cache TinyStories + tokenizer locally")
    prep.add_argument("--tokenizer-model", default="gpt2")
    prep.add_argument("--cache-dir", default=str(default_cache_dir()))
    prep.add_argument("--offline", action="store_true", help="Reuse-only (no network)")

    return parser


def _parse_layer_keep_ratios(raw: str | None) -> tuple[float, ...] | None:
    """Parse comma-separated keep ratios into a tuple, or None."""
    if raw is None:
        return None
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if not parts:
        return None
    return tuple(float(p) for p in parts)


def _cmd_prepare(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    if args.offline:
        # In offline mode, just validate that files are present by attempting local-only loads.
        from transbench.datasets import make_sampler

        _ = make_sampler(
            "tinystories",
            cache_dir=cache_dir,
            tokenizer_model=str(args.tokenizer_model),
            seed=0,
            offline=True,
        )
    else:
        prepare_tinystories(cache_dir=cache_dir, tokenizer_model=str(args.tokenizer_model))
    print(f"Prepared TinyStories + tokenizer in: {cache_dir}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    reports_cfg = ReportsConfig(reports_dir=Path(args.reports_dir))
    cache_dir = Path(args.cache_dir)

    def has_fla() -> bool:
        try:
            import fla.layers  # type: ignore

            return True
        except Exception:
            return False

    def fixed_name_for(cfg: BenchmarkConfig) -> str:
        dev = (cfg.device or "auto").replace(":", "-")
        dt = (cfg.dtype or "auto").replace(":", "-")
        kv = f"kv{cfg.num_kv_heads}" if cfg.num_kv_heads is not None else "kv-"
        ds = (cfg.dataset or "synthetic").replace(":", "-")

        extra = ""
        if (cfg.arch or "").lower() == "mig":
            lam = float(getattr(cfg, "mig_lambda", 0.0))
            gd = int(getattr(cfg, "mig_gate_dim", 64))
            kr = float(getattr(cfg, "mig_keep_ratio", 0.7))
            lam_s = (f"{lam:.6g}").replace(".", "p").replace("-", "m")
            kr_s = (f"{kr:.6g}").replace(".", "p").replace("-", "m")
            extra = f"_migG{gd}_lam{lam_s}_keep{kr_s}"
        elif (cfg.arch or "").lower() == "sil":
            rules = int(getattr(cfg, "sil_num_latent_rules", 64))
            temp = float(getattr(cfg, "sil_temperature", 1.0))
            ht = bool(getattr(cfg, "sil_hard_train", True))
            he = bool(getattr(cfg, "sil_hard_eval", True))
            temp_s = (f"{temp:.6g}").replace(".", "p").replace("-", "m")
            extra = f"_silR{rules}_T{temp_s}_ht{int(ht)}_he{int(he)}"
        elif (cfg.arch or "").lower() == "asr":
            noise = float(getattr(cfg, "asr_noise_std", 0.01))
            lam = float(getattr(cfg, "asr_lambda", 0.0))
            noise_s = (f"{noise:.6g}").replace(".", "p").replace("-", "m")
            lam_s = (f"{lam:.6g}").replace(".", "p").replace("-", "m")
            extra = f"_asrN{noise_s}_lam{lam_s}"
        return (
            f"{cfg.arch}_{ds}_L{cfg.num_layers}_H{cfg.hidden_size}_"
            f"S{cfg.seq_len}_B{cfg.batch_size}_"
            f"heads{cfg.num_heads}_{kv}_"
            f"{dev}_{dt}{extra}.json"
        )

    def run_one(arch: str) -> int:
        bench_cfg = BenchmarkConfig(
            arch=arch,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            ffn_mult=args.ffn_mult,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            vocab_size=args.vocab_size,
            initializer_range=args.initializer_range,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            warmup=args.warmup,
            steps=args.steps,
            learning_rate=args.lr,
            min_lr=args.min_lr,
            dataset=args.dataset,
            seed=args.seed,
            cache_dir=cache_dir,
            offline=bool(args.offline),
            device=args.device,
            dtype=args.dtype,
            tokenizer_model=args.tokenizer_model,
            mig_gate_dim=int(args.mig_gate_dim),
            mig_lambda=float(args.mig_lambda),
            mig_keep_ratio=float(args.mig_keep_ratio),
            mig_layer_keep_ratios=_parse_layer_keep_ratios(args.mig_layer_keep_ratios),
            poison_ratio=float(args.poison_ratio),
            sil_num_latent_rules=int(args.sil_num_rules),
            sil_temperature=float(args.sil_temperature),
            sil_hard_train=bool(args.sil_hard_train),
            sil_hard_eval=bool(args.sil_hard_eval),
            asr_noise_std=float(args.asr_noise_std),
            asr_lambda=float(args.asr_lambda),
            sil_lambda=float(args.sil_lambda),
        )

        result = run_benchmark(bench_cfg)
        meta = RunMeta(
            name=args.run_name,
            tags=list(args.tag or []),
            notes=args.notes,
            include_raw=bool(args.raw),
        )

        fixed_filename = fixed_name_for(bench_cfg) if args.replace else None
        report_path = write_report_with_meta(
            reports_cfg,
            bench_cfg,
            result,
            meta,
            fixed_filename=fixed_filename,
        )
        print(str(report_path))
        return 0

    def run_one_all_mode(arch: str) -> int:
        """When using --all, prefer stable filenames so we can skip missing-only runs."""

        bench_cfg = BenchmarkConfig(
            arch=arch,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            ffn_mult=args.ffn_mult,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            vocab_size=args.vocab_size,
            initializer_range=args.initializer_range,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            warmup=args.warmup,
            steps=args.steps,
            learning_rate=args.lr,
            min_lr=args.min_lr,
            dataset=args.dataset,
            seed=args.seed,
            cache_dir=cache_dir,
            offline=bool(args.offline),
            device=args.device,
            dtype=args.dtype,
            tokenizer_model=args.tokenizer_model,
            mig_gate_dim=int(args.mig_gate_dim),
            mig_lambda=float(args.mig_lambda),
            mig_keep_ratio=float(args.mig_keep_ratio),
            mig_layer_keep_ratios=_parse_layer_keep_ratios(args.mig_layer_keep_ratios),
            poison_ratio=float(args.poison_ratio),
            sil_num_latent_rules=int(args.sil_num_rules),
            sil_temperature=float(args.sil_temperature),
            sil_hard_train=bool(args.sil_hard_train),
            sil_hard_eval=bool(args.sil_hard_eval),
            asr_noise_std=float(args.asr_noise_std),
            asr_lambda=float(args.asr_lambda),
            sil_lambda=float(args.sil_lambda),
        )

        fixed_filename = fixed_name_for(bench_cfg)
        target = reports_cfg.reports_dir / fixed_filename
        if target.exists() and not args.replace:
            print(f"[skip] {arch}: already exists ({fixed_filename})")
            return 0

        result = run_benchmark(bench_cfg)
        meta = RunMeta(
            name=args.run_name,
            tags=list(args.tag or []),
            notes=args.notes,
            include_raw=bool(args.raw),
        )
        report_path = write_report_with_meta(
            reports_cfg,
            bench_cfg,
            result,
            meta,
            fixed_filename=fixed_filename,
        )
        print(str(report_path))
        return 0

    if args.all:
        # Always-available architectures (no optional deps)
        archs = ["gqa", "mig", "sil", "asr", "mhla"]
        # Optional mixins: only include if dependency is installed and the run
        # is targeting CUDA (these kernels typically don't support pure CPU).
        wants_cuda = args.device is None or str(args.device).startswith("cuda")
        if has_fla() and wants_cuda:
            archs.extend(["mamba2", "rwkv6", "retnet"])

        optional_archs = {"mamba2", "rwkv6", "retnet"}

        exit_code = 0
        for arch in archs:
            try:
                run_one_all_mode(arch)
            except ImportError as e:
                print(f"[skip] {arch}: {e}")
            except Exception as e:
                # Optional mixins can legitimately fail depending on the local
                # environment (compiler/CUDA/Triton availability). Don't fail the
                # whole `--all` run for these; report and continue.
                label = "skip" if arch in optional_archs else "error"
                print(f"[{label}] {arch}: {e}")
                if arch not in optional_archs:
                    exit_code = 1
        return exit_code

    return run_one(args.arch)


def _cmd_suite(args: argparse.Namespace) -> int:
    suite_path = Path(args.config)
    entries = load_suite_toml(suite_path)
    reports_cfg = ReportsConfig(reports_dir=Path(args.reports_dir))
    paths = run_suite(entries, reports_cfg, run_benchmark)
    for p in paths:
        print(str(p))
    return 0


def _cmd_make_manifest(args: argparse.Namespace) -> int:
    from transbench.reporting import rebuild_manifest

    reports_cfg = ReportsConfig(reports_dir=Path(args.reports_dir))
    manifest_path = rebuild_manifest(reports_cfg)
    print(str(manifest_path))
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    root = Path.cwd()
    do_reports = bool(args.reports or args.all or (not args.reports and not args.caches and not args.runs and not args.all))
    do_caches = bool(args.caches or args.all or (not args.reports and not args.caches and not args.runs and not args.all))
    do_runs = bool(args.runs or args.all)

    plan = build_clean_plan(root, reports=do_reports, caches=do_caches, runs=do_runs)

    if args.dry_run:
        for p in plan.files:
            print(f"[file] {p}")
        for p in plan.dirs:
            print(f"[dir]  {p}")
        return 0

    if not args.yes:
        total = len(plan.files) + len(plan.dirs)
        if total == 0:
            print("Nothing to clean")
            return 0
        resp = input(f"Delete {len(plan.files)} files and {len(plan.dirs)} directories? [y/N] ").strip().lower()
        if resp not in {"y", "yes"}:
            print("Aborted")
            return 1

    apply_clean_plan(plan)
    print("Clean complete")
    return 0


def _cmd_serve_dashboard(args: argparse.Namespace) -> int:
    import http.server
    import socketserver
    import errno

    root = Path.cwd()
    handler = http.server.SimpleHTTPRequestHandler

    try:
        with socketserver.TCPServer((args.host, args.port), handler) as httpd:
            url = f"http://{args.host}:{args.port}/dashboard/index.html"
            print(f"Serving from: {root}")
            print(f"Dashboard: {url}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                return 0
    except OSError as e:
        if getattr(e, "errno", None) in {errno.EADDRINUSE, 10048}:
            print(f"Port {args.port} is already in use. Stop the other server or choose --port.")
            return 2
        raise
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv(Path.cwd() / ".env")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "suite":
        return _cmd_suite(args)
    if args.command == "make-manifest":
        return _cmd_make_manifest(args)
    if args.command == "serve-dashboard":
        return _cmd_serve_dashboard(args)
    if args.command == "clean":
        return _cmd_clean(args)
    if args.command == "prepare":
        return _cmd_prepare(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
