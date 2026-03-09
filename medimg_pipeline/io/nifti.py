from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from medimg_pipeline.core.volume import VolumeData


def _extract_spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))


def load_nifti(path: str | Path) -> VolumeData:
    path = Path(path)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    spacing = _extract_spacing_from_affine(affine)

    return VolumeData(
        array=data,
        spacing=spacing,
        affine=affine,
        source_type="nifti",
        extra_metadata={"path": str(path)},
    )


def save_nifti_mask(mask: np.ndarray, reference: VolumeData, out_path: str | Path) -> None:
    out_path = Path(out_path)
    affine = reference.affine if reference.affine is not None else np.eye(4)
    img = nib.Nifti1Image(mask.astype(np.uint8), affine=affine)
    nib.save(img, str(out_path))
