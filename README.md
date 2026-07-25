# Visual Cortex fMRI Analysis 🧠

## Overview
This repository provides a robust, containerized pipeline for analyzing functional Magnetic Resonance Imaging (fMRI) data, with a primary focus on the visual cortex. 

Rather than treating the brain as a static, hardwired system, this project approaches neural activity as a highly dynamic and adaptable network. By extracting spatial features and voxel-based patterns from NIfTI (`.nii`) files, this pipeline aims to decode how visual stimuli are processed, mapped, and dynamically rewired in the brain.

## Tech Stack
* **Neuroimaging:** `nilearn`, `nibabel` (for 3D/4D spatial mapping)
* **Machine Learning:** `scikit-learn`, `numpy` (for voxel-level feature extraction)
* **Environment:** `Docker` (for reproducible, cloud-ready execution)

## Getting Started
Build the isolated Docker environment to run the pipeline without installing local dependencies:

```bash
docker build -t visual-cortex-env .
docker run -it -v $(pwd):/workspace visual-cortex-env
```
