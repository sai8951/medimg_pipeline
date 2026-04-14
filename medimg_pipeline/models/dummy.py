from __future__ import annotations

import numpy as np


class DummyInferencer:
    """
    Simple deterministic inferencer for testing.

    Strategy:
    - Uses intensity thresholding as a fake "model"
    - Produces stable outputs for debugging pipeline
    """

    def __init__(self, weights=None, device: str = "cpu"):
        # weights / device are ignored but kept for interface compatibility
        self.device = device

    def predict(self, volume):
        """
        Args:
            volume: Volume object with `.data` attribute (numpy array)

        Returns:
            np.ndarray (float32) same shape as input
        """
        data = volume.data.astype(np.float32)

        # normalize to [0,1] (robust)
        if data.max() > data.min():
            norm = (data - data.min()) / (data.max() - data.min())
        else:
            norm = np.zeros_like(data)

        # fake prediction: smooth-ish mask
        pred = norm

        return pred.astype(np.float32)
