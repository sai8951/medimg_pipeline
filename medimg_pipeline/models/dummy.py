from __future__ import annotations

import numpy as np


class DummyInferencer:
    """
    Simple deterministic inferencer for testing.

    Strategy:
    - Uses normalized voxel intensity as a fake prediction
    - Produces stable outputs for debugging pipeline behavior
    """

    def __init__(self, weights=None, device: str = "cpu"):
        # weights / device are ignored for now, but kept for interface compatibility
        self.device = device

    def predict(self, volume) -> np.ndarray:
        """
        Args:
            volume:
                VolumeData-like object with `.array` attribute

        Returns:
            np.ndarray:
                float32 prediction array with the same shape as input
        """
        arr = volume.array.astype(np.float32)

        vmin = float(arr.min())
        vmax = float(arr.max())

        if vmax > vmin:
            pred = (arr - vmin) / (vmax - vmin)
        else:
            pred = np.zeros_like(arr, dtype=np.float32)

        return pred.astype(np.float32)
