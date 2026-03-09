from __future__ import annotations

from medimg_pipeline.core.volume import VolumeData


def reorient_volume(volume: VolumeData, target_orientation: str = "RAS") -> VolumeData:
    # TODO: MVPではstub。必要になったら nibabel orientation utilities 等で実装
    volume.extra_metadata["requested_orientation"] = target_orientation
    return volume
