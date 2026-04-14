from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)


def make_dummy_dicom_series(
    out_dir: str = "../data/dicom_dummy",
    num_slices: int = 16,
    rows: int = 64,
    cols: int = 64,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    for i in range(num_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        filename = out_path / f"slice_{i:03d}.dcm"

        ds = FileDataset(
            str(filename),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # Patient / study / series
        ds.PatientName = "Dummy^Patient"
        ds.PatientID = "DUMMY001"
        ds.Modality = "CT"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyDate = date_str
        ds.StudyTime = time_str
        ds.SeriesNumber = 1
        ds.InstanceNumber = i + 1

        # Geometry
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i)]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.SpacingBetweenSlices = 1.0

        # Pixel data attributes
        ds.Rows = rows
        ds.Columns = cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0  # unsigned
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15

        # Optional CT-ish tags
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.WindowCenter = 500
        ds.WindowWidth = 1000

        arr = (np.random.rand(rows, cols) * 1000).astype(np.uint16)
        ds.PixelData = arr.tobytes()

        ds.save_as(str(filename), write_like_original=False)

    print(f"Saved DICOM series to: {out_path.resolve()}")


if __name__ == "__main__":
    make_dummy_dicom_series()
