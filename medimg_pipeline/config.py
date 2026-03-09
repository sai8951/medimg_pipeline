from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    type: Literal["nifti", "dicom"]
    path: str
    pattern: str | None = None


class OutputConfig(BaseModel):
    dir: str = "./results"
    save_mask: bool = True
    save_overlay: bool = True
    overlay_slices: list[str] = Field(default_factory=lambda: ["center"])


class PreprocessConfig(BaseModel):
    orientation: str = "RAS"
    resample_spacing: tuple[float, float, float] | None = None
    intensity_clip: tuple[float, float] | None = None
    normalize: Literal["zscore", "minmax", "none"] = "zscore"


class ModelConfig(BaseModel):
    type: Literal["dummy", "pytorch", "monai", "nnunet"] = "dummy"
    weights: str | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class InferenceConfig(BaseModel):
    mode: Literal["full_volume", "patch"] = "full_volume"
    batch_size: int = 1


class PostprocessConfig(BaseModel):
    threshold: float = 0.5
    keep_largest_component: bool = False


class ExportConfig(BaseModel):
    mask_format: Literal["nifti"] = "nifti"
    summary_csv: bool = True


class PipelineConfig(BaseModel):
    input: InputConfig
    output: OutputConfig = OutputConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    export: ExportConfig = ExportConfig()


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
