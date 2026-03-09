from __future__ import annotations

import numpy as np

from medimg_pipeline.core.volume import VolumeData


def apply_intensity_clip(volume: VolumeData, clip_range: tuple[float, float] | None) -> VolumeData:
    if clip_range is None:
        return volume

    vmin, vmax = clip_range
    volume.array = np.clip(volume.array, vmin, vmax)
    return volume


def normalize_volume(volume: VolumeData, method: str) -> VolumeData:
    if method == "none":
        return volume

    arr = volume.array.astype(np.float32)

    if method == "zscore":
        mean = float(arr.mean())
        std = float(arr.std())
        if std > 0:
            arr = (arr - mean) / std
    elif method == "minmax":
        amin = float(arr.min())
        amax = float(arr.max())
        if amax > amin:
            arr = (arr - amin) / (amax - amin)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    volume.array = arr
    return volume
