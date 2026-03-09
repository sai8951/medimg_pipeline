# [WIP] medimg-pipeline

Lightweight **YAML-configured CLI** for reproducible **3D medical image inference pipelines**.

Supports **NIfTI** and **DICOM** inputs with standardized preprocessing, model inference, and export of segmentation results.

**Keywords:** medical imaging · DICOM · NIfTI · deep learning · segmentation · volumetric data

---

## Overview

**medimg-pipeline** is a lightweight CLI tool for running reproducible 3D medical image inference pipelines using YAML configuration.

It supports:

- **NIfTI** and **DICOM** input formats
- standardized preprocessing
- PyTorch-based model inference
- segmentation export
- visualization overlays
- batch inference workflows

The tool is designed for researchers and engineers working with **MRI, CT, or PET volumetric data** who need clean and reproducible inference workflows.

---

## Demo

*(example images will be added here)*

Example output:

MRI slice + segmentation mask overlay.

## Installation (conda)

Create environment

```bash
git clone https://github.com/sai8951/medimg-pipeline
cd medimg-pipeline
conda env create -f environment.yml
conda activate med-pipe
pip install -e .
```

## Quick Start


### 1. Dry Run
Inspect inputs and configuration before running inference:

```bash
medimg-pipeline dry-run config/config_nifti.yaml
```

Example output:

```bash
Found 2 input(s)
[1] case001.nii.gz | shape=(128,256,256) | spacing=(1.0,1.0,1.0)
[2] case002.nii.gz | shape=(128,256,256) | spacing=(1.0,1.0,1.0)
```

### 2. Batch inference

```bash
medimg-pipeline run config/config_nifti.yaml
```

Input directory:

```text
data/
    case001.nii.gz
    case002.nii.gz
```

Output:

```text
results/
    case001_mask.nii.gz
    case002_mask.nii.gz
    summary.csv
```

## Features

- YAML-configured inference pipelines
- Support for **NIfTI** and **DICOM** input data
- Standardized preprocessing for volumetric medical images
- PyTorch-based model inference
- Segmentation mask export (NIfTI)
- Visualization overlays (PNG)
- Batch inference support
- Reproducible pipeline configuration
- Dry-run input inspection

## Example Configuration

```yaml
input:
  type: nifti
  path: ./data
  pattern: "*.nii.gz"

output:
  dir: ./results
  save_mask: true
  save_overlay: true

preprocess:
  orientation: RAS
  resample_spacing: [1.0, 1.0, 1.0]
  normalize: zscore

model:
  type: pytorch
  weights: ./weights/model.pt
  device: auto  # cpu, cuda, mps, auto

inference:
  mode: full_volume
  batch_size: 1

postprocess:
  threshold: 0.5
  keep_largest_component: false

export:
  mask_format: nifti
  summary_csv: true
```

## Supported Input Formats

- **NIfTI** (`.nii`, `.nii.gz`)
- **DICOM series**

DICOM directories containing a single imaging series are supported.  
Multi-series directories can be handled via configuration options.

## Outputs

- Segmentation mask (`.nii.gz`)
- Visualization overlays (`.png`)
- Summary table (`.csv`)

## Intended Use

This tool is intended for research workflows in medical image analysis:
- MRI segmentation
- CT organ segmentation
- PET/MRI analysis pipelines

## License
Apache License 2.0

## Citation
If you use this tool in your research, please cite this repository.
