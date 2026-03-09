from __future__ import annotations

from medimg_pipeline.core.volume import VolumeData


def resample_volume(
    volume: VolumeData,
    target_spacing: tuple[float, float, float] | None,
) -> VolumeData:
    # TODO: MVPではstub。将来 scipy / SimpleITK / MONAI で実装
    if target_spacing is None:
        return volume

    # TODO: actual resampling
    volume.extra_metadata["requested_resample_spacing"] = target_spacing
    return volume
