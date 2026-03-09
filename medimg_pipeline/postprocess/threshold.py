from __future__ import annotations

import numpy as np


def apply_threshold(prediction: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prediction >= threshold).astype(np.uint8)
