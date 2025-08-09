"""
Utilities for project-wide reproducibility.

Call set_global_seed(seed) at the start of any script or experiment to
ensure deterministic behavior across common libraries.
"""

from __future__ import annotations

import os
import random
from typing import Optional


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set seeds for Python, NumPy, and (optionally) PyTorch.

    Args:
        seed: Seed value to apply across libraries.
        deterministic_torch: If True, configures cuDNN for deterministic behavior.
    """
    # Python hash seed and stdlib random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    # PyTorch (optional)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - env dependent
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        pass

    # Silence noisy parallelism warnings in tokenizers when used
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


__all__ = ["set_global_seed"]


