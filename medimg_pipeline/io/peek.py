from __future__ import annotations

from pathlib import Path

import nibabel as nib
import pydicom


def peek_nifti(path: Path):
    img = nib.load(str(path))

    return {
        "shape": img.shape,
        "spacing": tuple(float(x) for x in img.header.get_zooms()[:3]),
    }


def peek_dicom(path: Path):
    """
    Assumes directory of DICOM files.
    Reads only first slice.
    """
    files = sorted(path.glob("*"))

    if not files:
        raise RuntimeError(f"No DICOM files found in {path}")

    ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)

    spacing = None

    if hasattr(ds, "PixelSpacing") and hasattr(ds, "SliceThickness"):
        spacing = (
            float(ds.PixelSpacing[0]),
            float(ds.PixelSpacing[1]),
            float(ds.SliceThickness),
        )

    return {
        "shape": ("unknown",),  # intentionally not loading full volume
        "spacing": spacing,
    }
