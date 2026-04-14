from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class InputConfig(BaseModel):
    type: Literal["nifti", "dicom"]
    path: str
    pattern: str | None = None


class OutputConfig(BaseModel):
    dir: str = "./results"
    save_mask: bool = True
    save_overlay: bool = True
    overlay_slices: list[str] = Field(default_factory=lambda: ["center"])

    @field_validator("overlay_slices")
    @classmethod
    def validate_overlay_slices(cls, v: list[str]) -> list[str]:
        allowed = {"center"}
        invalid = [x for x in v if x not in allowed]
        if invalid:
            raise ValueError(
                f"Unsupported overlay_slices={invalid}. Currently only ['center'] is supported."
            )
        return v


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

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return v


class PostprocessConfig(BaseModel):
    threshold: float = 0.5
    keep_largest_component: bool = False

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v


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

    @model_validator(mode="after")
    def validate_cross_section(self) -> "PipelineConfig":
        if self.inference.mode != "full_volume":
            raise ValueError(
                "inference.mode='patch' is not implemented yet. Use 'full_volume'."
            )

        if self.model.type in {"pytorch", "monai", "nnunet"} and not self.model.weights:
            raise ValueError(
                f"model.weights is required when model.type='{self.model.type}'."
            )

        return self


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
