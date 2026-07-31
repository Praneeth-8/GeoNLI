# GeoNLI VLM Pipeline

This folder contains a small local pipeline for testing `Qwen2-VL` on remote sensing imagery. The current setup is aimed at captioning and visual question answering on satellite or aerial tiles.

## Files

- `setup_env.bat` - creates the Windows environment and installs dependencies
- `download_model.py` - downloads the selected Qwen2-VL model into a local cache
- `test_inference.py` - runs captioning or VQA on an input image
- `scripts/download_vrsbench.py` - downloads VRSBench annotations and optional image archives
- `scripts/fetch_real_test_samples.py` - downloads a few demo images for quick manual testing

## Setup

```bat
cd C:\Manthan\GeoNLI\vlm-pipeline
setup_env.bat
python download_model.py
```

## Run inference

```bat
python test_inference.py
python test_inference.py --image path\to\tile.png
python test_inference.py --image path\to\tile.png --vqa "How many buildings are visible?"
```

If no image is passed, the script uses `data/sample_tile.png`.

## Notes

- `venv/`, `models/`, and downloaded datasets are not meant to be committed.
- The repository keeps a lightweight sample image for smoke testing, not full datasets or model weights.
- Real evaluation should be done against VRSBench or another remote sensing benchmark, not the synthetic sample alone.
