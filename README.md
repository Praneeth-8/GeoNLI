# GeoNLI

This repository contains the GeoNLI research notes, the active detection, tiling, and grounding code, a simple working frontend app, and a local vision-language pipeline for remote-sensing imagery.

## Contents

- `GeoNLI.pdf` - problem statement
- `Learning material links.pdf` - supporting references
- `main.py`, `detector.py`, `tiler.py`, `grounding.py` - continuation project code for detection, tiling, and text-guided grounding experiments
- `frontend/` - browser UI for uploading an image and viewing detection, grounding, captioning, and VQA outputs
- `webapp.py` - lightweight local web server that connects the frontend to the GeoNLI pipeline
- `vlm-pipeline/` - local Qwen2-VL setup for captioning and VQA
- `test result/` - retained testing reports, sample outputs, and evaluation helpers

## Research focus

The project explores how pretrained models can be combined into a pipeline for satellite-image reasoning. The main topics are:

- Vision Language Models
- Visual Question Answering
- Object Detection
- Image Tiling

One early direction used direct image-to-answer prompting. The selected direction adds object detection and tiling first, then passes structured context into the vision-language model.

## Detection and grounding flow

The root pipeline now supports two complementary steps:

- YOLO OBB detection over image tiles for object proposals
- optional Grounding DINO text-query grounding over the full image

Example:

```bash
python main.py --image path/to/image.png --query "house" --show
```

If `--query` is omitted, the script runs only the YOLO tiling pipeline.

## Quick Start & Setup

Follow these steps to run the complete GeoNLI pipeline and web interface from scratch:

### 1. Environment Setup

#### Windows (Automatic Setup):
```bat
cd vlm-pipeline
setup_env.bat
```

#### Manual Setup (Windows / Linux):
```bash
python -m venv vlm-pipeline/venv

# Activate virtual environment
# Windows:
vlm-pipeline\venv\Scripts\activate
# Linux/macOS:
# source vlm-pipeline/venv/bin/activate

# Install PyTorch with CUDA (adjust CUDA version if needed)
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 2. Download Models

The pipeline utilizes **YOLO11-OBB**, **Grounding DINO**, and **Qwen2-VL-2B-Instruct**.
- YOLO11-OBB and Grounding DINO weights download automatically on first run.
- To pre-cache the Vision-Language Model (Qwen2-VL):
```bash
python vlm-pipeline/download_model.py
```

### 3. Launch the Web Application

Start the backend server:
```bat
vlm-pipeline\venv\Scripts\python.exe webapp.py
```
*(or simply `python webapp.py` inside the activated virtual environment)*

Then open your browser at:
```text
http://127.0.0.1:8081
```

### 4. CLI Inference

Run CLI experiments directly:
```bash
# YOLO OBB + Tiling
python main.py --image path/to/image.png --show

# YOLO OBB + Tiling + Grounding DINO
python main.py --image path/to/image.png --query "solar panels" --show

# Qwen2-VL Direct Inference (Captioning / VQA)
python test_inference.py --image path/to/image.png --prompt "Describe the satellite image in detail."
```

## References

- ISRO GeoNLI problem statement
- Hugging Face Transformers documentation
- Qwen2-VL documentation
- Ultralytics YOLO documentation
