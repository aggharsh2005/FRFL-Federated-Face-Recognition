"""Small shared utilities for FRFL scripts."""
from pathlib import Path
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_directories(*paths) -> None:
    """Create any supplied output directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
