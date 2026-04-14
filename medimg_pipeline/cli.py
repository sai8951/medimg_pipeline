from __future__ import annotations

import typer
import yaml
from pydantic import ValidationError

from medimg_pipeline.config import load_config
from medimg_pipeline.pipeline import inspect_inputs, run_pipeline

app = typer.Typer(help="YAML-configured CLI for 3D medical image inference pipelines.")


@app.command()
def run(config_path: str) -> None:
    """Run inference pipeline from YAML config."""
    config = load_config(config_path)
    summaries = run_pipeline(config)

    typer.echo(f"Pipeline completed. Processed {len(summaries)} case(s).")
    for summary in summaries:
        typer.echo(
            f"- {summary['case_id']} | "
            f"shape={summary['shape']} | "
            f"spacing={summary['spacing']} | "
            f"mask_voxels={summary['mask_voxels']}"
        )


@app.command()
def validate(config_path: str) -> None:
    """Validate YAML config."""
    try:
        _ = load_config(config_path)
        typer.echo("Config is valid.")
    except (ValidationError, ValueError) as e:
        typer.echo("Config validation failed.")
        typer.echo(str(e))
        raise typer.Exit(code=1)


@app.command("dry-run")
def dry_run(config_path: str) -> None:
    """Inspect inputs and configuration without running inference."""
    config = load_config(config_path)

    typer.echo("Inspecting inputs...")
    typer.echo("Note: current dry-run loads full volumes to inspect shape/spacing.")
    typer.echo("")

    rows = inspect_inputs(config)

    typer.echo(f"Found {len(rows)} input(s).")
    for i, row in enumerate(rows, start=1):
        typer.echo(
            f"[{i}] {row['case_id']} | {row['input_path']} | "
            f"shape={row['shape']} | spacing={row['spacing']}"
        )

    typer.echo("")
    typer.echo("Preprocess")
    typer.echo(f"- orientation: {config.preprocess.orientation}")
    typer.echo(f"- resample_spacing: {config.preprocess.resample_spacing}")
    typer.echo(f"- intensity_clip: {config.preprocess.intensity_clip}")
    typer.echo(f"- normalize: {config.preprocess.normalize}")

    typer.echo("")
    typer.echo("Model")
    typer.echo(f"- type: {config.model.type}")
    typer.echo(f"- weights: {config.model.weights}")
    typer.echo(f"- device: {config.model.device}")

    typer.echo("")
    typer.echo("Inference")
    typer.echo(f"- mode: {config.inference.mode}")
    typer.echo(f"- batch_size: {config.inference.batch_size}")

    typer.echo("")
    typer.echo("Output")
    typer.echo(f"- dir: {config.output.dir}")
    typer.echo(f"- save_mask: {config.output.save_mask}")
    typer.echo(f"- save_overlay: {config.output.save_overlay}")
    typer.echo(f"- overlay_slices: {config.output.overlay_slices}")

    typer.echo("")
    typer.echo("Export")
    typer.echo(f"- mask_format: {config.export.mask_format}")
    typer.echo(f"- summary_csv: {config.export.summary_csv}")


@app.command("example-config")
def example_config(input_type: str = "nifti") -> None:
    """Print example config to stdout."""
    if input_type == "dicom":
        example = {
            "input": {
                "type": "dicom",
                "path": "./data/case001",
            },
            "output": {
                "dir": "./results",
                "save_mask": True,
                "save_overlay": True,
                "overlay_slices": ["center"],
            },
            "preprocess": {
                "orientation": "RAS",
                "resample_spacing": [1.0, 1.0, 1.0],
                "intensity_clip": [-1000, 1000],
                "normalize": "zscore",
            },
            "model": {
                "type": "dummy",
                "weights": None,
                "device": "cpu",
            },
            "inference": {
                "mode": "full_volume",
                "batch_size": 1,
            },
            "postprocess": {
                "threshold": 0.5,
                "keep_largest_component": False,
            },
            "export": {
                "mask_format": "nifti",
                "summary_csv": True,
            },
        }
    else:
        example = {
            "input": {
                "type": "nifti",
                "path": "./data",
                "pattern": "*.nii.gz",
            },
            "output": {
                "dir": "./results",
                "save_mask": True,
                "save_overlay": True,
                "overlay_slices": ["center"],
            },
            "preprocess": {
                "orientation": "RAS",
                "resample_spacing": [1.0, 1.0, 1.0],
                "intensity_clip": None,
                "normalize": "zscore",
            },
            "model": {
                "type": "dummy",
                "weights": None,
                "device": "cpu",
            },
            "inference": {
                "mode": "full_volume",
                "batch_size": 1,
            },
            "postprocess": {
                "threshold": 0.5,
                "keep_largest_component": False,
            },
            "export": {
                "mask_format": "nifti",
                "summary_csv": True,
            },
        }

    typer.echo(yaml.safe_dump(example, sort_keys=False))
