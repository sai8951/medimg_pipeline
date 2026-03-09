from __future__ import annotations

import numpy as np


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    # TODO: MVPではstub, 将来的に scipy.ndimage.label 等で実装
    return mask.astype(np.uint8)
