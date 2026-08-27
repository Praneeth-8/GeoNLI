# Testing Notes

These are lightweight notes from local checks done on July 24, 2026.

## Environment

- Python 3.11
- PyTorch 2.4.1 with CUDA 12.1
- GPU: NVIDIA GeForce GTX 1650 Ti with 4 GB VRAM
- Model used: `Qwen/Qwen2-VL-2B-Instruct`

## What was checked

1. The helper scripts load and compile without syntax errors.
2. The pipeline can target a small local image for captioning or VQA.
3. The 2B model is the realistic option for a 4 GB GPU.

## Practical observations

- 7B variants are not a good fit on this machine.
- The 2B model with quantization is the usable local configuration.
- Latency is noticeable on consumer hardware, especially for larger images.

## Caveats

- These notes are not a benchmark report.
- The synthetic sample image is only a smoke test.
- Proper evaluation should use held-out remote sensing data such as VRSBench.
