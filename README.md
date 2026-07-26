# GeoNLI 

## Overview

This file contains the development carried out for the GeoNLI module for satellite image analysis.

The work includes integrating an object detection pipeline and vision language model to answer natural language questions about satellite imagery.


---

# Research Objective

The primary objective of this research was to investigate how existing pretrained AI models can be combined into a single pipeline capable of solving the GeoNLI problem.

The work concentrated on understanding

- Vision Language Models
- Visual Question Answering
- Object Detection
- Image Tiling.


---
# References

- ISRO GeoNLI Problem Statement
- Hugging Face Transformers Documentation
- Qwen2.5-VL Documentation
- Ultralytics YOLO Documentation

---



During the research, multiple approaches for building a VQA system were studied.

## Approach 1 — Direct Vision Language Model

```
Image
      │
      ▼
 Vision Language Model
      │
      ▼
    Answer
```

In this approach only the image and user question are provided to the VLM.

Advantages

- Simple implementation
- Minimal preprocessing

Limitations

- Small objects may be missed
- Difficult to reason over large satellite images
- High-resolution imagery may exceed model limitations

---

## Approach 2 — Object Detection Assisted VQA (Selected)

```
Satellite Image
        │
        ▼
Image Tiling
        │
        ▼
YOLO Detection
        │
        ▼
Detection Summary
        │
        ▼
Prompt Builder
        │
        ▼
Qwen2.5-VL
        │
        ▼
Answer
```

This approach combines computer vision with a Vision Language Model.

YOLO performs object detection while Qwen performs natural language reasoning.

This architecture was selected because it provides structured information to the language model instead of relying entirely on visual inference.

---

# Vision Language Models

The research focused on understanding how modern Vision Language Models perform reasoning.

Unlike conventional CNN-based classifiers, Vision Language Models jointly process

- images
- text
- user instructions

within a single model.

Several open-source models were investigated including

- Qwen2.5-VL
- LLaVA
- Hugging Face multimodal models

Qwen2.5-VL was selected for the prototype because it integrates directly with the Hugging Face Transformers ecosystem and supports conversational image understanding.

---

# Object Detection

A Vision Language Model is capable of recognising many objects directly from an image.

However, satellite imagery introduces several challenges

- extremely high resolution
- very small objects
- overlapping structures

To improve reliability, the proposed pipeline first performs explicit object detection using YOLO.

The detector returns

- object class
- confidence score
- location

These detections become structured context for the language model.

---

# Image Tiling

Large satellite images cannot always be processed efficiently as a single image.

The investigated solution divides the original image into smaller tiles or smaller overlapping tiles before inference.

Advantages

- Detects smaller objects
- Reduces memory usage
- Improves object detection accuracy

Overlapping tiles may produce duplicate detections.

This requires a post-processing stage before the detections are forwarded to the VQA module.

---

# Prompts for VLM

A Vision Language Model does not receive only a user question.

Instead, it receives structured instructions describing its task.

The model uses both

- the original image
- structured textual context
to generate the final response.

Passing every YOLO detection to the Vision Language Model produces unnecessarily large prompts.
Different questions require different contextual information. Examples include counting questions, existence questions or description questions.
External geographic reasoning questions can done by integrating GIS data.

---





