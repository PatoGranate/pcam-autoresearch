"""Small helpers: seeding, device selection, run folders."""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator we use, so runs are repeatable.

    Note: on GPU, results are very close but not always bit-identical between
    runs, because some CUDA kernels use non-deterministic algorithms for speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)           # also seeds CUDA / MPS
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own reproducible seed (for augmentations)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Prefer an NVIDIA GPU (CUDA), then Apple Silicon (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(run_name: str, root: str = "runs") -> Path:
    """Create runs/<run_name>_<timestamp>/ so re-running never overwrites results."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
