from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from medimg_pipeline.core.volume import VolumeData


def _get_center_slice_index(volume: VolumeData) -> int:
    return volume.array.shape[0] // 2


def save_center_overlay_png(
    volume: VolumeData,
    mask: np.ndarray,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    z = _get_center_slice_index(volume)

    image_slice = volume.array[z]
    mask_slice = mask[z]

    plt.figure()
    plt.imshow(image_slice, cmap="gray")
    plt.imshow(mask_slice, alpha=0.4)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
