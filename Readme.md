# GeoNLI

**GeoNLI** is a vision-language pipeline for detecting objects in satellite imagery and answering natural-language questions about the image.

The project combines computer vision and vision-language models to identify objects, locate regions of interest, and generate image-grounded responses.

## Overview

GeoNLI aims to bridge object detection and natural-language understanding for satellite imagery.

Given a satellite image and a user query, the pipeline processes the image, identifies relevant objects or regions, and uses visual-language reasoning to generate a response.

## Pipeline

1. **Image Tiling** – Divides large satellite images into smaller overlapping tiles for processing.
2. **Object Detection** – Uses YOLO OBB to detect and localize objects.
3. **Detection Merging** – Maps tile-level detections back to the original image and merges overlapping detections using geometric operations.
4. **Grounding** – Uses Grounding DINO to identify image regions corresponding to text queries.
5. **Visual Question Answering** – Uses Qwen2-VL to interpret image content and answer natural-language questions.

## Tech Stack

- Python
- PyTorch
- YOLO OBB (Ultralytics)
- Grounding DINO
- Qwen2-VL
- OpenCV
- Shapely
- Hugging Face Transformers

## Objective

To explore the integration of object detection, visual grounding, and vision-language models for intelligent analysis of satellite imagery.

# GeoNLI Setup

## 1. Create Virtual Environment

```bash
cd ~/Project/GeoNLI
python3.12 -m venv venv312
source venv312/bin/activate
pip install --upgrade pip
```

## 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
--index-url https://download.pytorch.org/whl/cu121
```

## 3. Install Dependencies

Comment out PyTorch lines in `requirements.txt`, then:
If not commented
```bash
pip install -r requirements.txt
```

## 4. Verify Installation

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```
## 5. Download the Model

```bash
cd vlm-pipline/
python3 download_model.py
```

This downloads the VLM model weights.his downloads the vlm's weight
## 6. Run

```bash
source venv312/bin/activate
python3 webapp.py
```
