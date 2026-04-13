from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# NOTE: Project direction
# TinyStories is the primary real-language dataset. Adversarial datasets
# (poisoned_needle, sparse_signal) test structural advantages. Utility
# generators (zeros, ramp) exist only for hardware diagnostics / smoke tests.

DIAGNOSTIC_DATASETS: set[str] = {"zeros", "zero", "ramp", "arange"}
ADVERSARIAL_DATASETS: set[str] = {"poisoned_needle", "sparse_signal"}


TINY_STORIES_DATASET_ID = "roneneldan/TinyStories"
TINY_STORIES_SPLIT = "train"
TINY_STORIES_TEXT_FIELD = "text"


def list_dataset_ids() -> list[str]:
    """Return canonical dataset IDs supported by the CLI."""

    return ["tinystories", "poisoned_needle", "sparse_signal", "zeros", "ramp"]


def is_supported_dataset(dataset: str) -> bool:
    d = (dataset or "").lower().strip()
    return d == "tinystories" or d in DIAGNOSTIC_DATASETS or d in ADVERSARIAL_DATASETS


def default_cache_dir() -> Path:
    # Prefer a user cache, not the repo.
    return Path.home() / ".cache" / "transbench" / "datasets"


def _require_hf_train_deps() -> tuple[Any, Any]:
    """Import optional training deps used for TinyStories sampling."""

    try:
        import datasets  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "TinyStories sampling requires the optional dependency 'datasets'. "
            "Install it with: uv pip install -e '.[train]'"
        ) from e

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "TinyStories sampling requires the optional dependency 'transformers'. "
            "Install it with: uv pip install -e '.[train]'"
        ) from e

    return datasets, AutoTokenizer


def _load_tokenizer(AutoTokenizer: Any, *, model: str, cache_dir: Path, offline: bool) -> Any:
    # Prefer slow tokenizers when available to avoid downloading large `tokenizer.json`
    # files (fast tokenizers) in flaky environments.
    kwargs = {
        "cache_dir": str(cache_dir),
        "local_files_only": bool(offline),
    }
    try:
        return AutoTokenizer.from_pretrained(model, use_fast=False, **kwargs)
    except Exception:
        return AutoTokenizer.from_pretrained(model, use_fast=True, **kwargs)


def prepare_tinystories(*, cache_dir: Path | None = None, tokenizer_model: str = "gpt2") -> None:
    """Download/cache TinyStories and the tokenizer locally.

    HF Datasets + Transformers already cache by default, but this forces the
    one-time download step so repeated benchmark runs won't need network.
    """

    cache_dir = cache_dir or default_cache_dir()
    datasets, AutoTokenizer = _require_hf_train_deps()

    # Tokenizer files
    _ = _load_tokenizer(AutoTokenizer, model=str(tokenizer_model), cache_dir=cache_dir, offline=False)

    # Dataset files
    builder = datasets.load_dataset_builder(TINY_STORIES_DATASET_ID, cache_dir=str(cache_dir))
    builder.download_and_prepare()


def sample_input_ids(
    dataset: str,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    cache_dir: Path | None = None,
    tokenizer_model: str | None = None,
    offline: bool = False,
) -> np.ndarray:
    """Return a (batch_size, seq_len) numpy array of token IDs.

    For diagnostic datasets (zeros, ramp) and adversarial datasets (sparse_signal, poisoned_needle),
    no external dependencies are required.
    For TinyStories, uses HF Datasets + Transformers tokenizer (optional dependencies).
    """

    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be > 0")

    d = (dataset or "tinystories").lower().strip()

    if d in {"zeros", "zero"}:
        return np.zeros((batch_size, seq_len), dtype=np.int64)

    if d in {"ramp", "arange"}:
        row = np.remainder(np.arange(seq_len, dtype=np.int64), int(vocab_size)).astype(np.int64, copy=False)
        return np.repeat(row[None, :], batch_size, axis=0)

    if d == "poisoned_needle":
        # Poisoned Needle: start from random base data then inject poison
        # into center. Works without optional [train] deps.
        rng = np.random.default_rng(int(seed))
        base_ids = rng.integers(0, int(vocab_size), size=(batch_size, seq_len), dtype=np.int64)
        return _inject_poison(base_ids, poison_ratio=0.85, vocab_size=int(vocab_size), rng=rng)

    if d == "sparse_signal":
        rng = np.random.default_rng(int(seed))
        return _generate_sparse_signal(
            batch_size=batch_size, seq_len=seq_len, vocab_size=int(vocab_size),
            signal_ratio=0.15, motif_len=8, rng=rng,
        )

    if d != "tinystories":
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Choose one of: {', '.join(list_dataset_ids())}"
        )

    datasets, AutoTokenizer = _require_hf_train_deps()
    cache_dir = cache_dir or default_cache_dir()

    # Tokenizer model defaults to something small-ish if caller doesn't care.
    # Users can pass the original notebook tokenizer/model id explicitly.
    tok_model = (tokenizer_model or "gpt2").strip()
    tokenizer = AutoTokenizer.from_pretrained(
        tok_model,
        cache_dir=str(cache_dir),
        local_files_only=bool(offline),
    )
    # Re-load using the slow-tokenizer preference (above call keeps backward compatibility
    # for some tokenizers; this ensures we consistently avoid fast-tokenizer downloads).
    tokenizer = _load_tokenizer(AutoTokenizer, model=tok_model, cache_dir=cache_dir, offline=bool(offline))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    download_config = None
    if bool(offline):
        try:
            download_config = datasets.DownloadConfig(local_files_only=True)
        except Exception:
            download_config = None

    ds = datasets.load_dataset(
        TINY_STORIES_DATASET_ID,
        split=TINY_STORIES_SPLIT,
        cache_dir=str(cache_dir),
        download_config=download_config,
    )

    rng = np.random.default_rng(int(seed))
    idxs = rng.integers(0, len(ds), size=int(batch_size), dtype=np.int64)
    texts: list[str] = []
    for i in idxs.tolist():
        row = ds[int(i)]
        text = row.get(TINY_STORIES_TEXT_FIELD) if isinstance(row, dict) else None
        texts.append(str(text or ""))

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=int(seq_len),
        return_tensors="np",
    )
    ids = enc["input_ids"].astype(np.int64, copy=False)

    # Ensure IDs fit the configured vocab_size if user is training a scratch model.
    # If vocab_size matches tokenizer vocab, this is a no-op.
    if int(vocab_size) > 0:
        ids = np.remainder(ids, int(vocab_size)).astype(np.int64, copy=False)
    return ids


class _TinyStoriesSampler:
    def __init__(self, *, cache_dir: Path, tokenizer_model: str, seed: int, offline: bool):
        datasets, AutoTokenizer = _require_hf_train_deps()

        self._rng = np.random.default_rng(int(seed))
        self._cache_dir = cache_dir
        self._tokenizer_model = tokenizer_model

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            cache_dir=str(cache_dir),
            local_files_only=bool(offline),
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"

        download_config = None
        if bool(offline):
            try:
                download_config = datasets.DownloadConfig(local_files_only=True)
            except Exception:
                download_config = None

        self._ds = datasets.load_dataset(
            TINY_STORIES_DATASET_ID,
            split=TINY_STORIES_SPLIT,
            cache_dir=str(cache_dir),
            download_config=download_config,
        )

    def __call__(self, *, batch_size: int, seq_len: int, vocab_size: int) -> np.ndarray:
        idxs = self._rng.integers(0, len(self._ds), size=int(batch_size), dtype=np.int64)
        texts: list[str] = []
        for i in idxs.tolist():
            row = self._ds[int(i)]
            text = row.get(TINY_STORIES_TEXT_FIELD) if isinstance(row, dict) else None
            texts.append(str(text or ""))

        enc = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=int(seq_len),
            return_tensors="np",
        )
        ids = enc["input_ids"].astype(np.int64, copy=False)
        if int(vocab_size) > 0:
            ids = np.remainder(ids, int(vocab_size)).astype(np.int64, copy=False)
        return ids


def _inject_poison(
    token_ids: np.ndarray,
    *,
    poison_ratio: float = 0.85,
    vocab_size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Inject noise into the center of token sequences (Poisoned Needle).

    Preserves the first and last ~5% of each sequence as "needle" signal,
    then replaces *poison_ratio* of the center tokens with random IDs.

    Args:
        token_ids: ``(batch_size, seq_len)`` array of token IDs.
        poison_ratio: Fraction of center tokens to corrupt (0 < r <= 1).
        vocab_size: Upper bound for random token IDs.
        rng: NumPy random generator (deterministic if provided).

    Returns:
        A **copy** with poison injected; original is not mutated.
    """
    if rng is None:
        rng = np.random.default_rng()
    ids = token_ids.copy()
    _batch, seq_len = ids.shape
    edge = max(1, int(seq_len * 0.05))
    center_start = edge
    center_end = seq_len - edge
    center_len = center_end - center_start
    if center_len <= 0:
        return ids
    n_poison = max(1, int(center_len * float(poison_ratio)))
    for b in range(ids.shape[0]):
        positions = rng.choice(center_len, size=n_poison, replace=False) + center_start
        ids[b, positions] = rng.integers(0, int(vocab_size), size=n_poison, dtype=np.int64)
    return ids


def _generate_sparse_signal(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    signal_ratio: float = 0.15,
    motif_len: int = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate sequences with a repeating motif at evenly-spaced signal positions.

    Non-signal positions are filled with uniform random tokens.  The motif is
    deterministic for a given RNG state, so it is reproducible across calls.

    Args:
        batch_size: Number of sequences.
        seq_len: Length of each sequence.
        vocab_size: Upper bound for random token IDs.
        signal_ratio: Fraction of positions that carry the motif (0, 1].
        motif_len: Length of the repeating motif pattern.
        rng: NumPy random generator.

    Returns:
        ``(batch_size, seq_len)`` array of token IDs.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate a deterministic motif for this RNG state.
    motif = rng.integers(0, int(vocab_size), size=int(motif_len), dtype=np.int64)

    # Compute evenly-spaced signal positions.
    n_signal = max(1, int(seq_len * float(signal_ratio)))
    signal_positions = np.linspace(0, seq_len - 1, n_signal, dtype=np.int64)

    # Fill with random noise, then place motif at signal positions.
    ids = rng.integers(0, int(vocab_size), size=(batch_size, seq_len), dtype=np.int64)
    for i, pos in enumerate(signal_positions):
        ids[:, pos] = motif[i % len(motif)]
    return ids


def sparse_signal_positions(seq_len: int, signal_ratio: float) -> np.ndarray:
    """Return the signal-position indices for a sparse_signal sequence.

    Matches the positions used by ``_generate_sparse_signal`` so callers
    can compute gate selectivity on known signal vs noise positions.
    """
    n_signal = max(1, int(seq_len * float(signal_ratio)))
    return np.linspace(0, seq_len - 1, n_signal, dtype=np.int64)


def make_sampler(
    dataset: str,
    *,
    cache_dir: Path | None = None,
    tokenizer_model: str | None = None,
    seed: int = 0,
    offline: bool = False,
    poison_ratio: float = 0.85,
    signal_ratio: float = 0.15,
    motif_len: int = 8,
):
    """Create a callable sampler for datasets that benefit from caching.

    Supports ``tinystories``, ``poisoned_needle``, and ``sparse_signal``.
    """

    d = (dataset or "").lower().strip()
    if d not in {"tinystories", "poisoned_needle", "sparse_signal"}:
        raise ValueError(f"make_sampler only supports 'tinystories'/'poisoned_needle'/'sparse_signal' (got: {dataset})")

    if d == "sparse_signal":
        return _SparseSiganlSampler(
            signal_ratio=float(signal_ratio), motif_len=int(motif_len), seed=int(seed),
        )

    cache_dir = cache_dir or default_cache_dir()
    tok_model = (tokenizer_model or "gpt2").strip()
    base = _TinyStoriesSampler(cache_dir=cache_dir, tokenizer_model=tok_model, seed=int(seed), offline=bool(offline))
    if d == "poisoned_needle":
        return _PoisonedNeedleSampler(base=base, poison_ratio=float(poison_ratio), seed=int(seed))
    return base


class _PoisonedNeedleSampler:
    """Wraps a TinyStories sampler to inject poison into the center of sequences."""

    def __init__(self, *, base: _TinyStoriesSampler, poison_ratio: float, seed: int):
        self._base = base
        self._poison_ratio = float(poison_ratio)
        self._rng = np.random.default_rng(int(seed) + 9999)

    def __call__(self, *, batch_size: int, seq_len: int, vocab_size: int) -> np.ndarray:
        ids = self._base(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size)
        return _inject_poison(
            ids, poison_ratio=self._poison_ratio, vocab_size=int(vocab_size), rng=self._rng,
        )


class _SparseSiganlSampler:
    """Generates fresh sparse_signal batches with a fixed motif across calls."""

    def __init__(self, *, signal_ratio: float, motif_len: int, seed: int):
        self._signal_ratio = float(signal_ratio)
        self._motif_len = int(motif_len)
        self._rng = np.random.default_rng(int(seed))

    def __call__(self, *, batch_size: int, seq_len: int, vocab_size: int) -> np.ndarray:
        return _generate_sparse_signal(
            batch_size=batch_size, seq_len=seq_len, vocab_size=int(vocab_size),
            signal_ratio=self._signal_ratio, motif_len=self._motif_len, rng=self._rng,
        )
