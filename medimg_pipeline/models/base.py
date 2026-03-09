from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from medimg_pipeline.core.volume import VolumeData


class BaseInferencer(ABC):
    @abstractmethod
    def predict(self, volume: VolumeData) -> np.ndarray:
        raise NotImplementedError
