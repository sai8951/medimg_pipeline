from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from medimg_pipeline.core.volume import VolumeData


def load_dicom_series(path: str | Path) -> VolumeData:
    path = Path(path)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(path))
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {path}")

    series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(str(path), series_id)
    reader.SetFileNames(file_names)
    image = reader.Execute()

    array = sitk.GetArrayFromImage(image).astype(np.float32)  # (D, H, W)
    spacing_sitk = image.GetSpacing()  # usually (x, y, z)
    spacing = (float(spacing_sitk[2]), float(spacing_sitk[1]), float(spacing_sitk[0]))

    return VolumeData(
        array=array,
        spacing=spacing,
        origin=tuple(float(x) for x in image.GetOrigin()),
        direction=tuple(float(x) for x in image.GetDirection()),
        modality=None,
        patient_id=None,
        source_type="dicom",
        extra_metadata={
            "path": str(path),
            "series_id": series_id,
            "num_files": len(file_names),
        },
    )
