# medimg_pipeline
YAML-configured CLI for reproducible 3D medical image inference pipelines with NIfTI and DICOM support.

**Keywords:** medical imaging, DICOM, NIfTI, deep learning, segmentation, volumetric data

- **medimg_pipeline** is a lightweight CLI tool for running reproducible 3D medical image inference pipelines using YAML configuration.
- It supports both **NIfTI** and **DICOM** inputs, standardized preprocessing, model inference, and export of segmentation results and visualizations.
- The tool is designed for researchers and engineers working with MRI, CT, or PET volumetric data who need clean and reproducible inference workflows.

## TODO
Put some example snapshots here.  
MRI slice + segmentation mask overlay for example.

## Quick Example
Run an inference pipeline with a YAML configuration:

```bash
medimg-pipeline run config.yaml
```

Expected output:

```text
results/
├── case001_mask.nii.gz
├── case001_overlay.png
└── summary.csv
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

## Example Configuration

```yaml
input:
  type: dicom
  path: ./data/case001

output:
  dir: ./results
  save_overlay: true

preprocess:
  orientation: RAS
  resample_spacing: [1.0, 1.0, 1.0]
  normalize: zscore

model:
  type: pytorch
  weights: ./weights/model.pt
  device: cuda

inference:
  mode: full_volume

postprocess:
  threshold: 0.5
```

## Installation

```bash
pip install medimg-pipeline
```

or

```bash
git clone https://github.com/sai8951/medimg_pipeline
cd medimg-pipeline
pip install -e .
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

This tool is intended for research and development workflows in medical image analysis, including:
- MRI segmentation
- CT organ segmentation
- PET/MRI analysis pipelines

## License
Apache License 2.0

## Citation
If you use this tool in your research, please cite this repository.
