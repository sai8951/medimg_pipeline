from __future__ import annotations

import numpy as np
import torch

from medimg_pipeline.core.volume import VolumeData
from medimg_pipeline.models.base import BaseInferencer


def resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("device='cuda' was requested, but CUDA is not available.")

    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("device='mps' was requested, but MPS is not available.")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device}")


class PyTorchInferencer(BaseInferencer):
    def __init__(self, weights: str | None = None, device: str = "cpu") -> None:
        self.weights = weights
        self.device = resolve_torch_device(device)
        # TODO: load actual PyTorch model and move it to self.device

    def predict(self, volume: VolumeData) -> np.ndarray:
        # TODO: replace with real model inference
        arr = volume.array.astype(np.float32)
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            pred = (arr - arr_min) / (arr_max - arr_min)
        else:
            pred = np.zeros_like(arr, dtype=np.float32)
        return pred
