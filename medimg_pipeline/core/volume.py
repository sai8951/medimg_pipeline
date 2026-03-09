from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VolumeData:
    array: np.ndarray
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] | None = None
    direction: tuple[float, ...] | None = None
    affine: np.ndarray | None = None
    modality: str | None = None
    patient_id: str | None = None
    source_type: str = "unknown"
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape
