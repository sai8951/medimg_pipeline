from __future__ import annotations

from pathlib import Path

import pandas as pd

from medimg_pipeline.core.volume import VolumeData
from medimg_pipeline.io.nifti import save_nifti_mask


def ensure_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_mask_nifti(mask, reference: VolumeData, out_path: str | Path) -> None:
    save_nifti_mask(mask=mask, reference=reference, out_path=out_path)


def write_summary_csv(rows: list[dict], out_path: str | Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
