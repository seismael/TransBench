"""Quick CUDA smoke test – run a tiny 5-step benchmark on GPU."""
import sys
import warnings

warnings.filterwarnings("ignore")

sys.argv = [
    "transbench", "benchmark",
    "--arch", "gqa",
    "--dataset", "synthetic",
    "--num-layers", "2",
    "--hidden-size", "64",
    "--num-heads", "2",
    "--num-kv-heads", "1",
    "--vocab-size", "4096",
    "--seq-len", "32",
    "--batch-size", "2",
    "--warmup", "1",
    "--steps", "5",
    "--lr", "3e-4",
    "--device", "cuda",
    "--dtype", "float16",
    "--run-name", "smoke-cuda-fp16",
    "--reports-dir", "reports",
]

from transbench.cli import main  # noqa: E402

main()
