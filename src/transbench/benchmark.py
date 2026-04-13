from __future__ import annotations

import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

from transbench.datasets import default_cache_dir, is_supported_dataset, make_sampler, sample_input_ids
from transbench.reporting import BenchmarkResult, ModelBreakdown, SystemInfo


@dataclass(frozen=True)
class BenchmarkConfig:
    arch: str
    num_layers: int
    hidden_size: int
    ffn_mult: float
    num_heads: int
    num_kv_heads: int | None
    vocab_size: int
    initializer_range: float
    seq_len: int
    batch_size: int
    warmup: int
    steps: int
    learning_rate: float
    min_lr: float
    dataset: str
    seed: int | None
    cache_dir: Path | None
    offline: bool
    device: str | None
    dtype: str | None
    tokenizer_model: str
    mig_gate_dim: int
    mig_lambda: float
    sil_num_latent_rules: int
    sil_temperature: float
    sil_hard_train: bool
    sil_hard_eval: bool
    asr_noise_std: float
    asr_lambda: float
    mig_keep_ratio: float = 0.7
    mig_layer_keep_ratios: tuple[float, ...] | None = None
    poison_ratio: float = 0.85
    sil_lambda: float = 0.0


def _collect_mig_aux_loss(model: torch.nn.Module) -> torch.Tensor | None:
    aux_loss_total = 0.0
    num_mig_layers = 0
    
    for module in model.modules():
        fn = getattr(module, "mig_aux_loss", None)
        if callable(fn):
            try:
                val = fn()
            except Exception:
                val = None
            
            if isinstance(val, torch.Tensor):
                # Accumulate loss from each MIG layer (both Sparse and Dense)
                # Dense layers return 0.0, but we include them in the count 
                # so the lambda scaling is consistent across model depths.
                aux_loss_total = aux_loss_total + val
                num_mig_layers += 1

    if num_mig_layers == 0:
        return None
    
    # Normalize (Average over layers)
    return aux_loss_total / num_mig_layers


def _collect_asr_aux_loss(model: torch.nn.Module) -> torch.Tensor | None:
    terms: list[torch.Tensor] = []
    for module in model.modules():
        fn = getattr(module, "asr_aux_loss", None)
        if callable(fn):
            try:
                value = fn()
            except Exception:
                value = None
            if isinstance(value, torch.Tensor):
                terms.append(value)
    if not terms:
        return None
    return torch.stack(terms).mean()


def _collect_sil_aux_loss(model: torch.nn.Module) -> torch.Tensor | None:
    terms: list[torch.Tensor] = []
    for module in model.modules():
        fn = getattr(module, "sil_aux_loss", None)
        if callable(fn):
            try:
                value = fn()
            except Exception:
                value = None
            if isinstance(value, torch.Tensor):
                terms.append(value)
    if not terms:
        return None
    return torch.stack(terms).mean()


def _stable_seed(dataset: str, seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    # deterministic default based on dataset name
    return abs(hash(dataset)) % (2**31)


def _auto_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _auto_dtype(dtype: str | None, device: torch.device) -> torch.dtype:
    if dtype is None:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    normalized = dtype.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _build_model(cfg: BenchmarkConfig) -> torch.nn.Module:
    from transbench.modules.archi_modules import StackedMixinForCausalLM
    from transbench.modules.ffn_modules import FFN

    # Import mixins lazily so the file can be imported without heavy optional deps.
    from transbench.modules import mixin_modules as mixins

    arch = cfg.arch.lower()
    mixin_module = None

    use_fla = False
    if (cfg.device or "").lower().strip() == "cuda":
        try:
            import fla.layers  # type: ignore  # noqa: F401

            use_fla = True
        except Exception:
            # If FLA isn't available, keep going with the pure PyTorch fallback
            # implementations so CUDA benchmarking still works.
            use_fla = False

    if arch == "gqa":
        mixin_module = mixins.GroupedQuerySelfAttentionMixin(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_kv_heads,
        )
    elif arch == "rnn":
        mixin_module = mixins.RNNMixin(hidden_size=cfg.hidden_size)
    elif arch == "lstm":
        mixin_module = mixins.LSTMMixin(hidden_size=cfg.hidden_size)
    elif arch == "mamba2":
        mixin_module = mixins.Mamba2Mixin(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_kv_heads,
            use_fla=use_fla,
        )
    elif arch == "rwkv6":
        mixin_module = mixins.RWKV6Mixin(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_kv_heads,
            use_fla=use_fla,
        )
    elif arch in {"mhla", "mla"}:
        mixin_module = mixins.MultiHeadLatentAttentionMixin(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_kv_heads,
        )
    elif arch in {"retnet", "msr", "retention"}:
        mixin_module = mixins.MultiScaleRetentionMixin(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_kv_heads,
            use_fla=use_fla,
        )
    elif arch == "mig":
        from transbench.modules.mig_module import MIGAttention

        layer_keep_ratios = getattr(cfg, "mig_layer_keep_ratios", None)
        mixin_module = MIGAttention(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_kv_heads,
            gate_dim=int(getattr(cfg, "mig_gate_dim", 64)),
            keep_ratio=float(getattr(cfg, "mig_keep_ratio", 0.7)),
            layer_keep_ratios=list(layer_keep_ratios) if layer_keep_ratios else None,
        )
    elif arch == "sil":
        from transbench.modules.sil_module import StochasticInductionMixin

        mixin_module = StochasticInductionMixin(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_kv_heads,
            num_latent_rules=int(getattr(cfg, "sil_num_latent_rules", 64)),
            temperature=float(getattr(cfg, "sil_temperature", 1.0)),
            hard_train=bool(getattr(cfg, "sil_hard_train", True)),
            hard_eval=bool(getattr(cfg, "sil_hard_eval", True)),
        )
    elif arch == "asr":
        from transbench.modules.asr_module import ASRAttentionMixin

        mixin_module = ASRAttentionMixin(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_kv_heads,
            noise_std=float(getattr(cfg, "asr_noise_std", 0.05)),
        )
    else:
        raise ValueError(f"Unknown arch: {cfg.arch}")

    intermediate_size = int(cfg.hidden_size * cfg.ffn_mult)
    ffn_module = FFN(hidden_size=cfg.hidden_size, intermediate_size=intermediate_size)

    model = StackedMixinForCausalLM(
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        initializer_range=float(getattr(cfg, "initializer_range", 0.02)),
        embedding_module=None,
        final_norm_module=None,
        lm_head_module=None,
        mixin_module=mixin_module,
        ffn_module=ffn_module,
        positionnal_module=None,
        freeze_lm_modules=False,
        vocab_size=cfg.vocab_size,
    )

    return model


def _parameter_breakdown(model: torch.nn.Module) -> ModelBreakdown:
    # Use getattr + Any to avoid static typing false positives for dynamically
    # attached submodules on the composed model.
    m: Any = model
    total_parameters = sum(p.numel() for p in model.parameters())

    embedding_module = getattr(m, "embedding_module", None)
    embedding_parameters = (
        sum(p.numel() for p in embedding_module.parameters())
        if isinstance(embedding_module, torch.nn.Module)
        else 0
    )

    ffn_parameters = 0
    mixin_parameters = 0
    try:
        stacked = getattr(m, "stacked_mixin_block", None)
        layers = getattr(stacked, "layers", None)
        if layers is not None:
            for layer in layers:
                ffn = getattr(layer, "ffn_module", None)
                if isinstance(ffn, torch.nn.Module):
                    ffn_parameters += sum(p.numel() for p in ffn.parameters())
                mixin = getattr(layer, "mixin_module", None)
                if isinstance(mixin, torch.nn.Module):
                    mixin_parameters += sum(p.numel() for p in mixin.parameters())
    except Exception:
        pass

    lm_head_module = getattr(m, "lm_head_module", None)
    lm_head_parameters = (
        sum(p.numel() for p in lm_head_module.parameters())
        if isinstance(lm_head_module, torch.nn.Module)
        else 0
    )

    return ModelBreakdown(
        total_parameters=int(total_parameters),
        embedding_parameters=int(embedding_parameters),
        mixin_parameters=int(mixin_parameters),
        ffn_parameters=int(ffn_parameters),
        lm_head_parameters=int(lm_head_parameters),
    )


def _get_system_info(device: torch.device) -> SystemInfo:
    gpu_name = None
    cuda_version = None
    cuda_available = False
    # Only probe CUDA when we are actually running on CUDA.
    # This avoids noisy warnings on systems with unsupported GPUs.
    if device.type == "cuda":
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = None
            cuda_version = torch.version.cuda

    return SystemInfo(
        os=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_available=bool(cuda_available),
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        cpu_brand=platform.processor() or None,
        cpu_count=os.cpu_count() or 0,
        ram_gb=(float(psutil.virtual_memory().total / (1024**3)) if psutil is not None else 0.0),
        device=str(device),
    )


def run_benchmark(cfg: BenchmarkConfig) -> BenchmarkResult:
    device = _auto_device(cfg.device)
    dtype = _auto_dtype(cfg.dtype, device)

    model = _build_model(cfg)
    model.to(device=device)

    # keep layernorm etc in fp32? keep simple; user can tune later.
    if dtype != torch.float32:
        model = model.to(dtype=dtype)

    model.train()

    system = _get_system_info(device)
    breakdown = _parameter_breakdown(model)

    dataset = (cfg.dataset or "synthetic").lower().strip()
    if not is_supported_dataset(dataset):
        raise ValueError(
            f"Unsupported dataset '{cfg.dataset}'. Use 'tinystories' or synthetic/zeros/ramp."
        )

    cache_dir = Path(cfg.cache_dir).expanduser() if cfg.cache_dir is not None else default_cache_dir()
    offline = bool(cfg.offline)
    seed = _stable_seed(dataset, cfg.seed)

    # For TinyStories / poisoned_needle we sample fresh batches per step.
    # For synthetic/zeros/ramp we can just reuse a fixed batch.
    sampler = None
    if dataset in {"tinystories", "poisoned_needle"}:
        sampler = make_sampler(
            dataset,
            cache_dir=cache_dir,
            tokenizer_model=getattr(cfg, "tokenizer_model", "gpt2"),
            seed=int(seed),
            offline=offline,
            poison_ratio=float(getattr(cfg, "poison_ratio", 0.85)),
        )
        ids_np = sampler(batch_size=int(cfg.batch_size), seq_len=int(cfg.seq_len), vocab_size=int(cfg.vocab_size))
    else:
        ids_np = sample_input_ids(
            dataset,
            batch_size=int(cfg.batch_size),
            seq_len=int(cfg.seq_len),
            vocab_size=int(cfg.vocab_size),
            seed=int(seed),
            cache_dir=cache_dir,
            tokenizer_model=getattr(cfg, "tokenizer_model", "gpt2"),
            offline=offline,
        )

    input_ids = torch.from_numpy(ids_np).to(device=device, dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate))

    base_lr = float(cfg.learning_rate)
    min_lr = float(cfg.min_lr)
    warmup_steps = max(0, int(cfg.warmup))
    train_steps = max(1, int(cfg.steps))

    def lr_at(global_step: int) -> float:
        if warmup_steps > 0 and global_step < warmup_steps:
            return base_lr * float(global_step + 1) / float(warmup_steps)

        # Cosine decay over the measured training steps.
        # We treat "cfg.steps" as the post-warmup region.
        t = float(global_step - warmup_steps)
        T = float(max(1, train_steps))
        # Clamp to [0, 1]
        p = max(0.0, min(1.0, t / T))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))

    def set_lr(lr: float) -> None:
        for pg in optimizer.param_groups:
            pg["lr"] = float(lr)

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup (also records LR/loss for curves)
    loss_series: list[float] = []
    lr_series: list[float] = []
    aux_loss_series: list[float] = []
    asr_aux_loss_series: list[float] = []
    sil_aux_loss_series: list[float] = []
    total_loss_series: list[float] = []

    mig_lambda = float(getattr(cfg, "mig_lambda", 0.0))
    asr_lambda = float(getattr(cfg, "asr_lambda", 0.0))
    sil_lambda = float(getattr(cfg, "sil_lambda", 0.0))

    global_step = 0
    for _ in range(warmup_steps):
        if sampler is not None:
            ids_np = sampler(batch_size=int(cfg.batch_size), seq_len=int(cfg.seq_len), vocab_size=int(cfg.vocab_size))
            input_ids = torch.from_numpy(ids_np).to(device=device, dtype=torch.long)
        lr = lr_at(global_step)
        set_lr(lr)
        loss_ce = model(input_ids, return_loss=True)
        mig_aux = _collect_mig_aux_loss(model) if mig_lambda > 0.0 else None
        mig_aux_scaled = mig_aux * mig_lambda if mig_aux is not None else None
        asr_aux = _collect_asr_aux_loss(model) if asr_lambda > 0.0 else None
        asr_aux_scaled = asr_aux * asr_lambda if asr_aux is not None else None
        sil_aux = _collect_sil_aux_loss(model) if sil_lambda > 0.0 else None
        sil_aux_scaled = sil_aux * sil_lambda if sil_aux is not None else None

        loss_total = loss_ce
        if mig_aux_scaled is not None:
            loss_total = loss_total + mig_aux_scaled
        if asr_aux_scaled is not None:
            loss_total = loss_total + asr_aux_scaled
        if sil_aux_scaled is not None:
            loss_total = loss_total + sil_aux_scaled

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _sync()
        loss_series.append(float(loss_ce.detach().cpu()))
        lr_series.append(float(lr))
        if mig_aux_scaled is not None:
            aux_loss_series.append(float(mig_aux_scaled.detach().cpu()))
        if asr_aux_scaled is not None:
            asr_aux_loss_series.append(float(asr_aux_scaled.detach().cpu()))
        if sil_aux_scaled is not None:
            sil_aux_loss_series.append(float(sil_aux_scaled.detach().cpu()))
        if mig_aux_scaled is not None or asr_aux_scaled is not None or sil_aux_scaled is not None:
            total_loss_series.append(float(loss_total.detach().cpu()))
        global_step += 1

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    fwd_times = []
    train_times = []
    losses = []

    process = psutil.Process(os.getpid()) if psutil is not None else None
    rss_peak_mb = None

    # Measure forward time separately (no backward)
    # For long training runs, a long forward-only timing loop adds noticeable startup delay.
    # We cap it aggressively for large `--steps` while still producing a stable estimate.
    timing_cap = 200
    if train_steps >= 5000:
        timing_cap = 10
    elif train_steps >= 1000:
        timing_cap = 25
    timing_steps = int(min(timing_cap, max(1, train_steps)))

    try:
        dev_s = str(device)
    except Exception:
        dev_s = "device"
    print(f"[{cfg.arch} | {dev_s}] timing_forward_steps={timing_steps}", flush=True)
    model.eval()
    with torch.no_grad():
        for _ in range(timing_steps):
            t0 = time.perf_counter()
            _ = model(input_ids, return_loss=False)
            _sync()
            fwd_times.append(time.perf_counter() - t0)
            try:
                if process is not None:
                    rss_peak_mb = max(rss_peak_mb or 0.0, process.memory_info().rss / (1024**2))
            except Exception:
                pass

    # Measure train step time (loss + backward + step)
    model.train()
    print(f"[{cfg.arch} | {dev_s}] train_steps={train_steps}", flush=True)
    t_train_start = time.perf_counter()
    if train_steps >= 1000:
        progress_every = 100
    else:
        progress_every = max(1, train_steps // 10)
    for _ in range(train_steps):
        t0 = time.perf_counter()
        if sampler is not None:
            ids_np = sampler(batch_size=int(cfg.batch_size), seq_len=int(cfg.seq_len), vocab_size=int(cfg.vocab_size))
            input_ids = torch.from_numpy(ids_np).to(device=device, dtype=torch.long)
        lr = lr_at(global_step)
        set_lr(lr)
        loss_ce = model(input_ids, return_loss=True)
        mig_aux = _collect_mig_aux_loss(model) if mig_lambda > 0.0 else None
        mig_aux_scaled = mig_aux * mig_lambda if mig_aux is not None else None
        asr_aux = _collect_asr_aux_loss(model) if asr_lambda > 0.0 else None
        asr_aux_scaled = asr_aux * asr_lambda if asr_aux is not None else None
        sil_aux = _collect_sil_aux_loss(model) if sil_lambda > 0.0 else None
        sil_aux_scaled = sil_aux * sil_lambda if sil_aux is not None else None

        loss_total = loss_ce
        if mig_aux_scaled is not None:
            loss_total = loss_total + mig_aux_scaled
        if asr_aux_scaled is not None:
            loss_total = loss_total + asr_aux_scaled
        if sil_aux_scaled is not None:
            loss_total = loss_total + sil_aux_scaled

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _sync()
        train_times.append(time.perf_counter() - t0)
        loss_val = float(loss_ce.detach().cpu())
        losses.append(loss_val)
        loss_series.append(loss_val)
        lr_series.append(float(lr))
        if mig_aux_scaled is not None:
            aux_loss_series.append(float(mig_aux_scaled.detach().cpu()))
        if asr_aux_scaled is not None:
            asr_aux_loss_series.append(float(asr_aux_scaled.detach().cpu()))
        if sil_aux_scaled is not None:
            sil_aux_loss_series.append(float(sil_aux_scaled.detach().cpu()))
        if mig_aux_scaled is not None or asr_aux_scaled is not None or sil_aux_scaled is not None:
            total_loss_series.append(float(loss_total.detach().cpu()))
        global_step += 1
        step_idx = global_step - warmup_steps
        if step_idx <= 1 or step_idx == train_steps or (progress_every > 0 and step_idx % progress_every == 0):
            elapsed_s = time.perf_counter() - t_train_start
            try:
                dev_s = str(device)
            except Exception:
                dev_s = "device"
            print(
                f"[{cfg.arch} | {dev_s}] step {step_idx}/{train_steps} loss={loss_val:.5f} lr={lr:.3g} elapsed={elapsed_s:.1f}s",
                flush=True,
            )
        try:
            if process is not None:
                rss_peak_mb = max(rss_peak_mb or 0.0, process.memory_info().rss / (1024**2))
        except Exception:
            pass

    tokens_per_step = cfg.batch_size * cfg.seq_len
    mean_train_s = float(np.mean(train_times))
    tokens_per_s = float(tokens_per_step / mean_train_s) if mean_train_s > 0 else 0.0

    peak_mem_mb = None
    if device.type == "cuda":
        try:
            peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
        except Exception:
            peak_mem_mb = None
    else:
        peak_mem_mb = float(rss_peak_mb) if rss_peak_mb is not None else None

    return BenchmarkResult(
        system=system,
        model=breakdown,
        config={
            "arch": cfg.arch,
            "num_layers": cfg.num_layers,
            "hidden_size": cfg.hidden_size,
            "ffn_mult": cfg.ffn_mult,
            "num_heads": cfg.num_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "vocab_size": cfg.vocab_size,
            "initializer_range": float(getattr(cfg, "initializer_range", 0.02)),
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "warmup": int(cfg.warmup),
            "steps": int(cfg.steps),
            "learning_rate": float(cfg.learning_rate),
            "min_lr": float(cfg.min_lr),
            "dataset": dataset,
            "seed": int(seed),
            "cache_dir": str(cache_dir),
            "offline": bool(getattr(cfg, "offline", False)),
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "tokenizer_model": getattr(cfg, "tokenizer_model", "gpt2"),
            "mig_gate_dim": int(getattr(cfg, "mig_gate_dim", 64)),
            "mig_lambda": float(getattr(cfg, "mig_lambda", 0.0)),
            "mig_keep_ratio": float(getattr(cfg, "mig_keep_ratio", 0.7)),
            "sil_num_latent_rules": int(getattr(cfg, "sil_num_latent_rules", 64)),
            "sil_temperature": float(getattr(cfg, "sil_temperature", 1.0)),
            "sil_hard_train": bool(getattr(cfg, "sil_hard_train", True)),
            "sil_hard_eval": bool(getattr(cfg, "sil_hard_eval", True)),
            "asr_noise_std": float(getattr(cfg, "asr_noise_std", 0.05)),
            "asr_lambda": float(getattr(cfg, "asr_lambda", 0.0)),
            "sil_lambda": float(getattr(cfg, "sil_lambda", 0.0)),
        },
        metrics={
            "forward_ms_mean": float(np.mean(fwd_times) * 1000.0),
            "forward_ms_p50": float(np.percentile(fwd_times, 50) * 1000.0),
            "forward_ms_p95": float(np.percentile(fwd_times, 95) * 1000.0),
            "train_step_ms_mean": float(np.mean(train_times) * 1000.0),
            "train_step_ms_p50": float(np.percentile(train_times, 50) * 1000.0),
            "train_step_ms_p95": float(np.percentile(train_times, 95) * 1000.0),
            "tokens_per_s": tokens_per_s,
            "loss_mean": float(np.mean(losses)) if losses else None,
            "loss_series": loss_series,
            "lr_series": lr_series,
            "mig_aux_loss_mean": float(np.mean(aux_loss_series)) if aux_loss_series else None,
            "asr_aux_loss_mean": float(np.mean(asr_aux_loss_series)) if asr_aux_loss_series else None,
            "sil_aux_loss_mean": float(np.mean(sil_aux_loss_series)) if sil_aux_loss_series else None,
            "total_loss_mean": float(np.mean(total_loss_series)) if total_loss_series else None,
            "total_loss_series": total_loss_series if total_loss_series else None,
            "peak_mem_mb": peak_mem_mb,
        },
    )
