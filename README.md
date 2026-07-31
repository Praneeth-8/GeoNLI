# GeoNLI

This repository contains working notes and a local vision-language pipeline for the GeoNLI problem statement.

## Contents

- `GeoNLI.pdf` - problem statement
- `Learning material links.pdf` - supporting references
- `vlm-pipeline/` - local Qwen2-VL setup for captioning and VQA on remote sensing imagery

## VLM pipeline

The code in `vlm-pipeline/` is set up for local experiments on a Windows machine with NVIDIA GPU support. It includes:

- environment setup
- hardware check
- model download helper
- inference script for captioning and VQA
- dataset download helpers

Large local artifacts such as the virtual environment, model cache, and downloaded datasets are intentionally ignored and should be recreated locally when needed.
