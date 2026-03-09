from __future__ import annotations

from pathlib import Path

import numpy as np

from medimg_pipeline.config import PipelineConfig
from medimg_pipeline.core.exceptions import UnsupportedInputError
from medimg_pipeline.io.dicom import load_dicom_series
from medimg_pipeline.io.nifti import load_nifti
from medimg_pipeline.io.writers import ensure_output_dir, write_mask_nifti, write_summary_csv
from medimg_pipeline.models.monai import MonaiInferencer
from medimg_pipeline.models.nnunet import NnUNetInferencer
from medimg_pipeline.models.pytorch import PyTorchInferencer
from medimg_pipeline.postprocess.components import keep_largest_component
from medimg_pipeline.postprocess.threshold import apply_threshold
from medimg_pipeline.transforms.normalize import apply_intensity_clip, normalize_volume
from medimg_pipeline.transforms.orientation import reorient_volume
from medimg_pipeline.transforms.resample import resample_volume
from medimg_pipeline.visualize.overlay import save_center_overlay_png


def collect_inputs(config: PipelineConfig) -> list[Path]:
    """
    Collect input targets from config.

    NIfTI:
      - if path is a file -> single case
      - if path is a directory -> collect by glob pattern

    DICOM:
      - if path is a directory with subdirectories -> each subdirectory is treated as one case
      - if path has no subdirectories -> path itself is treated as one case
    """
    path = Path(config.input.path)

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if config.input.type == "nifti":
        if path.is_file():
            return [path]

        pattern = config.input.pattern or "*.nii.gz"
        inputs = sorted(path.glob(pattern))
        return [p for p in inputs if p.is_file()]

    if config.input.type == "dicom":
        if not path.is_dir():
            raise ValueError("DICOM input path must be a directory.")

        subdirs = sorted([p for p in path.iterdir() if p.is_dir()])
        if subdirs:
            return subdirs

        return [path]

    raise UnsupportedInputError(f"Unsupported input type: {config.input.type}")


def _load_single_input(input_path: Path, input_type: str):
    if input_type == "nifti":
        return load_nifti(input_path)
    if input_type == "dicom":
        return load_dicom_series(input_path)
    raise UnsupportedInputError(f"Unsupported input type: {input_type}")


def _build_inferencer(config: PipelineConfig):
    model_type = config.model.type

    if model_type in {"dummy", "pytorch"}:
        return PyTorchInferencer(weights=config.model.weights, device=config.model.device)
    if model_type == "monai":
        return MonaiInferencer(weights=config.model.weights, device=config.model.device)
    if model_type == "nnunet":
        return NnUNetInferencer(weights=config.model.weights, device=config.model.device)

    raise ValueError(f"Unsupported model type: {model_type}")


def _preprocess_volume(volume, config: PipelineConfig):
    volume = reorient_volume(volume, config.preprocess.orientation)
    volume = resample_volume(volume, config.preprocess.resample_spacing)
    volume = apply_intensity_clip(volume, config.preprocess.intensity_clip)
    volume = normalize_volume(volume, config.preprocess.normalize)
    return volume


def _case_id_from_input_path(input_path: Path, input_type: str) -> str:
    if input_type == "nifti":
        name = input_path.name
        if name.endswith(".nii.gz"):
            return name[:-7]
        return input_path.stem

    # DICOM directory case
    return input_path.name


def inspect_inputs(config: PipelineConfig) -> list[dict]:
    """
    Used by dry-run: inspect input files/directories without running inference.
    """
    inputs = collect_inputs(config)
    rows: list[dict] = []

    for input_path in inputs:
        volume = _load_single_input(input_path, config.input.type)
        rows.append(
            {
                "case_id": _case_id_from_input_path(input_path, config.input.type),
                "input_path": str(input_path),
                "source_type": volume.source_type,
                "shape": str(volume.shape),
                "spacing": str(volume.spacing),
            }
        )

    return rows


def run_single_case(input_path: Path, config: PipelineConfig, output_dir: Path, inferencer) -> dict:
    """
    Run the pipeline for one case and return summary dict.
    """
    volume = _load_single_input(input_path, config.input.type)
    volume = _preprocess_volume(volume, config)

    prediction = inferencer.predict(volume)
    mask = apply_threshold(prediction, config.postprocess.threshold)

    if config.postprocess.keep_largest_component:
        mask = keep_largest_component(mask)

    case_id = _case_id_from_input_path(input_path, config.input.type)

    mask_path = None
    overlay_path = None

    if config.output.save_mask:
        mask_path = output_dir / f"{case_id}_mask.nii.gz"
        write_mask_nifti(mask, reference=volume, out_path=mask_path)

    if config.output.save_overlay:
        overlay_path = output_dir / f"{case_id}_overlay.png"
        save_center_overlay_png(volume, mask, overlay_path)

    summary = {
        "case_id": case_id,
        "input_path": str(input_path),
        "input_type": config.input.type,
        "shape": str(volume.shape),
        "spacing": str(volume.spacing),
        "mask_voxels": int(np.sum(mask)),
        "mask_path": str(mask_path) if mask_path else "",
        "overlay_path": str(overlay_path) if overlay_path else "",
    }
    return summary


def run_pipeline(config: PipelineConfig) -> list[dict]:
    """
    Run the pipeline for all collected inputs.
    Returns a list of per-case summaries.
    """
    inputs = collect_inputs(config)
    output_dir = ensure_output_dir(config.output.dir)
    inferencer = _build_inferencer(config)

    summaries: list[dict] = []

    for input_path in inputs:
        summary = run_single_case(input_path, config, output_dir, inferencer)
        summaries.append(summary)

    if config.export.summary_csv:
        write_summary_csv(summaries, output_dir / "summary.csv")

    return summaries
