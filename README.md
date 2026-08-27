# GeoNLI

This repository contains the GeoNLI research notes and a local vision-language pipeline for remote-sensing imagery.

## Contents

- `GeoNLI.pdf` - problem statement
- `Learning material links.pdf` - supporting references
- `vlm-pipeline/` - local Qwen2-VL setup for captioning and VQA

## Research focus

The project explores how pretrained models can be combined into a pipeline for satellite-image reasoning. The main topics are:

- Vision Language Models
- Visual Question Answering
- Object Detection
- Image Tiling

One early direction used direct image-to-answer prompting. The selected direction adds object detection and tiling first, then passes structured context into the vision-language model.

## VLM pipeline

The code in `vlm-pipeline/` is set up for local experiments on a Windows machine with NVIDIA GPU support. It includes:

- environment setup
- hardware check
- model download helper
- inference script for captioning and VQA
- dataset download helpers

Large local artifacts such as the virtual environment, model cache, and downloaded datasets are intentionally ignored and should be recreated locally when needed.

## References

- ISRO GeoNLI problem statement
- Hugging Face Transformers documentation
- Qwen2-VL documentation
- Ultralytics YOLO documentation
