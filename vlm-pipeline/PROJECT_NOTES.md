# PROJECT_NOTES

This document covers the `vlm-pipeline/` folder only. It is meant to be a future-you reference: plain language first, technical detail second, with enough detail to rebuild the project from scratch.

## 1. Project Overview

This project is a small local pipeline for testing a vision-language model on satellite or aerial imagery. In practice, it loads one image, sends it together with a text prompt to the model, and prints back either a caption or a direct answer to a visual question.

The main problem it solves is satellite image captioning and VQA, which means: "look at this overhead image and describe it" or "look at this overhead image and answer a question about it." The model used here is `Qwen2-VL`, a model that can read both images and text at the same time.

High-level flow:

```text
image file + prompt
-> processor prepares image/text into model inputs
-> Qwen2-VL generates output tokens
-> tokens are decoded into plain text
-> caption or answer is printed to the terminal
```

## 2. Folder and File Structure

The tables below cover every tracked file in `vlm-pipeline/`, plus the runtime-generated folders this project expects to create while running.

### 2.1 Tracked Files

| Path | What it does | Why it exists separately | Depends on / Called by / Output | Key functions or sections |
| --- | --- | --- | --- | --- |
| `vlm-pipeline/README.md` | Gives the shortest possible explanation of what the folder is for and how to get started. It tells you the model, the main scripts, and the first commands to run. | A README exists so someone can understand the project without opening the Python code first. It is the front door of the folder. | Read by humans. Points to `setup_env.bat`, `download_model.py`, and `test_inference.py`. Produces no runtime output. | Sections: project summary, file list, setup commands, inference examples, notes. |
| `vlm-pipeline/requirements.txt` | Lists every Python package the project expects to install. It is the environment contract for the pipeline. | Separating dependencies from code makes the environment easier to reproduce and update. | Used by `setup_env.bat` and any manual `pip install` flow. Produces an installed Python environment, not a file. | Sections: core deep learning, transformers ecosystem, Qwen2-VL support, quantization, LoRA/fine-tuning, image processing, data utilities, evaluation metrics, progress display. |
| `vlm-pipeline/setup_env.bat` | Creates a Windows virtual environment and installs all dependencies needed for model download and inference. | Setup is separate because environment creation is a one-time machine task, not part of the model logic. It also keeps the runtime scripts cleaner. | Depends on `python`, `pip`, internet access, and Windows batch execution. Called manually by the user. Output is `venv/` plus installed packages. | Creates `venv`, activates it, upgrades `pip`, installs PyTorch CUDA wheels, installs Transformers/Qwen dependencies, installs `bitsandbytes`, installs image/utility/evaluation packages, checks CUDA, pauses at the end. |
| `vlm-pipeline/download_model.py` | Downloads the chosen Qwen2-VL model and its processor into a local cache. The "processor" is the helper that handles text tokenization and image preprocessing. | Separate download logic keeps model acquisition out of inference code. That way inference can assume the model already exists locally. | Depends on `transformers` and Hugging Face access. Called manually before inference. Output is the cached model under `vlm-pipeline/models/`. | `DEFAULT_MODEL`, `main()`; processor download step; model weight download step; success/error handling. |
| `vlm-pipeline/test_inference.py` | Loads the model and runs captioning or VQA on one image. If no image is supplied, it uses the bundled sample image or creates a placeholder image. | This is the main demo script, so it is kept separate from setup and downloads. It is the actual "run the model on an image" entry point. | Depends on `torch`, `transformers`, `PIL`, `qwen_vl_utils`, and the cached model files. Called manually by the user. Output is a printed caption or answer in the terminal. | `create_placeholder_image()`, `build_runtime_config()`, `load_model_and_processor()`, `run_inference()`, `main()`. |
| `vlm-pipeline/TESTING_REPORT.md` | Records a small local testing summary, including the machine used and the practical VRAM limitation. | Testing notes are kept separate from code so you can see what was verified without reading implementation details. | Read by humans only. No runtime output. Helps explain current status and hardware constraints. | Markdown sections: environment, checks, practical observations, caveats. |
| `vlm-pipeline/PHOTO_CAPTIONS.md` | Stores short human-written captions for the sample images used in local testing. | Keeping captions separate makes it easy to compare what a human thinks is in the image versus what the model says. | Read by humans and possibly by future evaluation scripts. No runtime output. | Table mapping each sample image file to a caption. |
| `vlm-pipeline/VRSBENCH_SAMPLE_CAPTIONS.md` | Records a sample caption result for one VRSBench image. | A separate note file makes it easy to preserve a one-off benchmark-style observation without mixing it into code. | Read by humans only. No runtime output. | Short markdown note with one image name and one output caption. |
| `vlm-pipeline/scripts/download_vrsbench.py` | Downloads VRSBench annotation files and optional image archives from Hugging Face, then extracts the zip files locally. | Dataset download logic is separate from inference so the model script stays focused on running predictions. This script is also easier to rerun when data is lost. | Depends on `huggingface_hub`, `json`, `zipfile`, and internet access. Called manually. Output is `data/vrsbench/` plus downloaded zip/json files. | `download_file()`, `extract_zip()`, `inspect_annotations()`, `main()`. |
| `vlm-pipeline/scripts/fetch_real_test_samples.py` | Downloads a few public image samples for quick manual testing. These are convenience images, not the full benchmark dataset. | It is separate because it is just a small test-data helper, not the real dataset downloader. | Depends on `requests` and internet access. Called manually. Output is `data/test_samples/*.jpg`. | `SAMPLE_IMAGES` mapping, `main()`. |
| `vlm-pipeline/data/sample_tile.png` | A small synthetic sample image used when no real image path is passed to `test_inference.py`. It is a 512x512 RGB PNG. | The sample image exists so the pipeline can be smoke-tested without downloading anything else. | Used by `test_inference.py` when `--image` is omitted. Output is an image file on disk. | No code inside this file; it is a binary asset. |

### 2.2 Runtime-Generated Folders

These folders are not tracked as source code, but they are part of the project workflow and matter for understanding where outputs go.

| Path | What it contains | How it is created | Why it exists |
| --- | --- | --- | --- |
| `vlm-pipeline/venv/` | The local Python virtual environment. | Created by `setup_env.bat` with `python -m venv venv`. | Keeps project packages isolated from the system Python install. |
| `vlm-pipeline/models/` | Cached Qwen2-VL model files and processor files from Hugging Face. | Created by `download_model.py` and used by `test_inference.py`. | Makes inference work locally without re-downloading the model every time. |
| `vlm-pipeline/data/test_samples/` | Downloaded sample images such as `airport.jpg`, `urban_city.jpg`, and others. | Created by `scripts/fetch_real_test_samples.py`. | Gives quick real-image inputs for manual testing. |
| `vlm-pipeline/data/vrsbench/` | Downloaded VRSBench annotations and extracted image folders. | Created by `scripts/download_vrsbench.py`. | Holds the benchmark data in a local, usable format. |

## 3. How Data Is Acquired and Structured

There are three data paths in this project:

1. The bundled synthetic smoke-test image.
2. The small public demo images downloaded for quick local testing.
3. The VRSBench dataset download used for real benchmark-style work.

### 3.1 Bundled smoke-test image

File:

```text
vlm-pipeline/data/sample_tile.png
```

What it is:

- A synthetic 512x512 RGB image.
- It is already in final form, so there is no conversion step.
- If it is missing, `test_inference.py` creates a replacement automatically with `create_placeholder_image()`.

Why it exists:

- It lets you test the whole pipeline without depending on the internet or a dataset download.
- It is a smoke test, meaning it checks that the code runs end-to-end, not that the model is accurate.

### 3.2 Quick demo images

Downloader:

```text
vlm-pipeline/scripts/fetch_real_test_samples.py
```

How they are downloaded:

```python
requests.get(url, timeout=15)
```

The script downloads these files into:

```text
vlm-pipeline/data/test_samples/
```

Current sample names:

```text
airport.jpg
port_harbor.jpg
urban_city.jpg
agricultural_fields.jpg
```

Raw data before processing:

- Each file is downloaded directly as a `.jpg`.
- There is no extra annotation file and no conversion step.
- These are just convenience images for manual checks, not the benchmark dataset.

What the data looks like after download:

- A flat folder of JPG files in `data/test_samples/`.
- Each file can be passed directly to `test_inference.py --image ...`.

### 3.3 VRSBench dataset

Dataset source:

```text
xiang709/VRSBench
```

Dataset hub URL:

```text
https://huggingface.co/datasets/xiang709/VRSBench
```

Downloader:

```text
vlm-pipeline/scripts/download_vrsbench.py
```

Exactly how it is downloaded:

```python
hf_hub_download(
    repo_id="xiang709/VRSBench",
    repo_type="dataset",
    filename=filename,
    local_dir=target_dir,
)
```

Files the script tries to download:

```text
VRSBench_EVAL_Cap.json
VRSBench_EVAL_vqa.json
VRSBench_EVAL_referring.json
VRSBench_train.json
Annotations_val.zip
Images_val.zip
Images_train.zip
Annotations_train.zip
```

Raw data before processing:

- JSON annotation files describing captioning, VQA, and referring tasks.
- ZIP archives containing validation and training images and annotations.
- The downloader expects items in the JSON files to contain fields like `image_id`, `id`, `caption`, or `conversations`, because that is what `inspect_annotations()` tries to print.

What the data looks like after processing:

- JSON files stay as JSON files in `data/vrsbench/`.
- ZIP files are extracted into folders such as:

```text
vlm-pipeline/data/vrsbench/annotations_val/
vlm-pipeline/data/vrsbench/images_val/
vlm-pipeline/data/vrsbench/annotations_train/
vlm-pipeline/data/vrsbench/images_train/
```

Why the conversion is necessary:

- ZIP archives are compressed containers, so they must be unpacked before other scripts can read the files easily.
- Extracted folders are easier to inspect, script against, and feed into future evaluation code.

How to regenerate if lost:

```bat
python scripts\download_vrsbench.py --mode annotations
python scripts\download_vrsbench.py --mode eval
python scripts\download_vrsbench.py --mode full
```

The output directory defaults to:

```text
vlm-pipeline/data/vrsbench/
```

### 3.4 Model files

Model source:

```text
Qwen/Qwen2-VL-2B-Instruct
```

Model hub URL:

```text
https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
```

Download script:

```text
vlm-pipeline/download_model.py
```

How it is downloaded:

```python
AutoProcessor.from_pretrained(...)
Qwen2VLForConditionalGeneration.from_pretrained(...)
```

Where it is stored:

```text
vlm-pipeline/models/
```

How to regenerate if lost:

```bat
python download_model.py
```

## 4. How the Model Works, Step by Step

### 4.1 Which model is used, and why

The project uses:

```text
Qwen/Qwen2-VL-2B-Instruct
```

Plain-language reason:

- It is a vision-language model, so it can look at an image and read a prompt at the same time.
- The `2B` version is much smaller than the larger variants, which matters because the local machine described in the testing notes has only 4 GB of VRAM.
- The `Instruct` version is tuned to follow human instructions, which makes it better for captioning and question answering than a raw base model.

### 4.2 The actual code path from input to output

The end-to-end flow lives in `vlm-pipeline/test_inference.py`.

1. `main()` parses command-line arguments.

```python
parser.add_argument("--image", ...)
parser.add_argument("--prompt", ...)
parser.add_argument("--vqa", ...)
parser.add_argument("--max-tokens", ...)
parser.add_argument("--load-in-4bit", ...)
parser.add_argument("--load-in-8bit", ...)
parser.add_argument("--no-quantization", ...)
```

2. If no image path is passed, the script uses the bundled smoke-test image.

```python
vlm-pipeline/data/sample_tile.png
```

If that file is missing, `create_placeholder_image()` creates a simple synthetic image so the script still has something to run on.

3. The script chooses a prompt.

- `--vqa` uses the VQA prompt template.
- `--prompt` uses your custom text.
- Otherwise it uses the default caption prompt.

4. `build_runtime_config()` checks whether CUDA is available and decides the runtime mode.

- On GPU, it prefers quantized loading unless you disable it.
- On CPU, it uses plain float32 loading.

5. `load_model_and_processor()` loads the processor and model from the local cache.

```python
AutoProcessor.from_pretrained(...)
Qwen2VLForConditionalGeneration.from_pretrained(...)
```

Important detail:

- `local_files_only=True` means inference will only use files already on disk.
- If the model is not cached yet, the script tells you to run `python download_model.py`.

6. `run_inference()` opens the image with Pillow and builds the message format expected by Qwen2-VL.

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }
]
```

7. The processor converts the image and prompt into model-ready tensors.

```python
text_input = processor.apply_chat_template(...)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(...)
```

Plain-language meaning:

- `apply_chat_template()` wraps the text in the conversation format the model expects.
- `process_vision_info()` turns the image reference into the visual input format the model expects.
- `processor(...)` turns all of that into tensors, which are numeric arrays the model can work with.

8. The inputs are moved to the same device as the model.

```python
inputs = inputs.to(device)
```

9. The model generates output tokens.

```python
output_ids = model.generate(...)
```

Important settings:

- `do_sample=False` makes generation deterministic.
- `max_new_tokens` limits how long the answer can get.
- `temperature=None` and `top_p=None` keep the output from using random sampling.

10. The new tokens are decoded back into text.

```python
output_text = processor.batch_decode(...)
```

11. `main()` prints the final answer between separator lines.

```text
============================================================
<model answer here>
============================================================
```

### 4.3 Quantization, explained simply

Quantization means storing the model with fewer bits so it uses less memory.

This project supports:

```text
4-bit quantization
8-bit quantization
no quantization
```

How the code chooses:

- If CUDA is available and you do not turn quantization off, the script defaults to 4-bit loading.
- `--load-in-8bit` forces 8-bit loading.
- `--load-in-4bit` forces 4-bit loading.
- `--no-quantization` disables quantization and uses float16 on GPU or float32 on CPU.

Why this matters:

- Smaller bit-widths use less VRAM, which makes a big model fit on a small GPU.
- The tradeoff is that quantization can slightly reduce precision, but it is usually worth it for local testing.

The 4-bit settings used in the code:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

Plain-language meanings:

- `nf4` is a 4-bit format designed to preserve quality better than a naive 4-bit scheme.
- `bnb_4bit_compute_dtype=torch.float16` means computation still happens in half precision where possible.
- `bnb_4bit_use_double_quant=True` compresses the quantization metadata too, saving more memory.

### 4.4 What the config values control

The project does not use a separate YAML or JSON config file. Instead, configuration is spread across:

1. `requirements.txt`
2. `setup_env.bat`
3. Command-line arguments in `test_inference.py`
4. Hard-coded defaults inside the Python scripts

Important runtime settings in `test_inference.py`:

| Setting | Where it appears | What it means |
| --- | --- | --- |
| `DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"` | Top of file | The default model to load from Hugging Face. |
| `DEFAULT_CACHE_DIR = .../models` | Top of file | Where model files are stored locally. |
| `min_pixels=256 * 28 * 28` | `AutoProcessor.from_pretrained(...)` | The lower bound for visual input size handling. In plain language, it stops the processor from shrinking images too much. |
| `max_pixels=512 * 28 * 28` | `AutoProcessor.from_pretrained(...)` | The upper bound for visual input size handling. In plain language, it keeps very large images from exploding memory use. |
| `local_files_only=True` | Model and processor loading | Forces offline/local loading only. |
| `trust_remote_code=True` | Model and processor loading | Allows the model repo to provide custom loading code if needed. |
| `device_map="auto"` | GPU loading | Lets Transformers place model parts on the available device automatically. |
| `torch_dtype=torch.float16` | GPU loading | Uses half precision on GPU to save memory. |
| `torch_dtype=torch.float32` | CPU loading | Uses normal precision on CPU. |
| `do_sample=False` | Generation | Makes answers deterministic instead of random. |
| `max_new_tokens` | CLI option | Caps how long the answer can be. |

## 5. How to Run Everything, From a Clean Machine

These steps assume a clean Windows machine and a shell in `C:\Manthan\GeoNLI`.

### 1. Move into the project folder and create the environment

```bat
cd C:\Manthan\GeoNLI\vlm-pipeline
setup_env.bat
```

What success looks like:

- The script creates `venv/` if it does not already exist.
- It prints CUDA verification details.
- It ends with `Setup complete!`.

What broken looks like:

- `python` not found.
- `pip install` failures.
- `torch.cuda.is_available()` prints `False` on a machine where you expected GPU support.

### 2. Download the model

```bat
python download_model.py
```

What success looks like:

- The script prints the model ID and cache directory.
- It says `Downloading processor...` and `Downloading model weights...`.
- It ends with `Download complete.`
- The folder `vlm-pipeline/models/` fills with Hugging Face cache files.

What broken looks like:

- `transformers is not installed. Run setup_env.bat first.`
- `Processor download failed: ...`
- `Model download failed: ...`

### 3. Run the basic inference smoke test

```bat
python test_inference.py
```

What success looks like:

- The script loads the model from the local cache.
- It prints `Running inference...`.
- It prints a line like `Generated <n> tokens in <time>s (<rate> tok/s)`.
- It prints a caption or description between separator lines.

What output to expect:

- The exact caption is not fixed because the model generates text.
- The important sign of success is a coherent image description, not a specific sentence.

What broken looks like:

- `Image not found: ...`
- `Processor loading failed: ...`
- `Model loading failed: ...`
- The script stalls before printing a result, which usually means the model is too large for the machine or the download is incomplete.

### 4. Run the baseline test on real images

First download the sample images:

```bat
python scripts\fetch_real_test_samples.py
```

Then run inference on one of them:

```bat
python test_inference.py --image data\test_samples\airport.jpg
```

Optional VQA example:

```bat
python test_inference.py --image data\test_samples\urban_city.jpg --vqa "What is the dominant scene type?"
```

What success looks like:

- Images appear in `vlm-pipeline/data/test_samples/`.
- The model returns a caption or a question answer based on the picture.

What broken looks like:

- `requests` errors or HTTP failures in the downloader.
- The image path does not exist.
- The answer is nonsense because the model is underpowered or the image is outside the model's comfort zone.

### 5. Regenerate dataset files if they are lost

Annotations only:

```bat
python scripts\download_vrsbench.py --mode annotations
```

Evaluation set with validation images:

```bat
python scripts\download_vrsbench.py --mode eval
```

Full dataset:

```bat
python scripts\download_vrsbench.py --mode full
```

What success looks like:

- Files download from Hugging Face.
- The script extracts zip archives into folders under `vlm-pipeline/data/vrsbench/`.
- It prints an annotation summary at the end.

What broken looks like:

- Hugging Face download failures.
- Zip extraction failures.
- Missing or malformed JSON files.

## 6. Current Status

### Implemented and documented

- Windows setup script for environment creation.
- Model download script for Qwen2-VL.
- Single-image inference script for captioning and VQA.
- VRSBench dataset download helper.
- Small demo-image downloader for manual testing.
- Bundled synthetic smoke-test image.
- Human-readable notes files for sample outputs and testing observations.

### Not yet done

- Training code for fine-tuning the model.
- A real evaluation script that computes metrics against VRSBench automatically.
- A batch benchmark runner over many images.
- A trained model checkpoint saved by this repo.
- Any web app, notebook UI, or API wrapper.
- A cross-platform setup script for Linux/macOS.

### Known issues and limitations

- The local testing notes say the machine has only 4 GB of VRAM, so large models are not practical there.
- The `2B` model is the realistic local option; `7B` variants are too heavy for that setup.
- `bitsandbytes` may not work cleanly on every Windows machine, so quantized loading can depend on the local setup.
- `setup_env.bat` installs CUDA 12.1 PyTorch wheels, so the machine needs a compatible NVIDIA environment for GPU support.
- If CUDA is not available, the code falls back to CPU, but inference will be much slower.
- The demo sample images are for smoke testing only; they are not enough for meaningful benchmark claims.

## 7. Glossary

| Term | Plain-language definition |
| --- | --- |
| `VLM` | A vision-language model that can understand both images and text. |
| `VQA` | Visual question answering: asking a question about an image and getting a text answer. |
| `captioning` | Turning an image into a short text description. |
| `inference` | Using a trained model to make a prediction or generate text, not training it. |
| `fine-tuning` | Training an existing model a little more on your own data so it gets better at your task. |
| `quantization` | Compressing model numbers so the model uses less memory and runs more easily on small hardware. |
| `4-bit quantization` | A very compact way of storing weights that saves memory the most but may lose a little quality. |
| `8-bit quantization` | A less aggressive compression method that uses more memory than 4-bit but is usually safer. |
| `NF4` | A special 4-bit format designed to preserve more quality than a simple 4-bit encoding. |
| `double quantization` | Compressing the quantization metadata too, which saves a bit more memory. |
| `VRAM` | Memory on the GPU; the main resource that limits how large a model you can load. |
| `processor` | The helper that prepares text and images for the model, including tokenization and resizing. |
| `tokenizer` | The part that turns text into numbered pieces the model can understand. |
| `prompt` | The text instruction you give the model, such as "describe this image." |
| `prompt template` | A fixed text pattern used to format the input in the way the model expects. |
| `device_map` | A setting that tells Transformers where to place the model, such as CPU or GPU. |
| `torch_dtype` | The numeric precision used for model weights and computation, such as float16 or float32. |
| `local_files_only` | A loading mode that uses only files already on disk and does not try to fetch from the internet. |
| `cache_dir` | The folder where downloaded model files are stored locally. |
| `generate()` | The model method that produces new text tokens one step at a time. |
| `max_new_tokens` | The maximum number of new tokens the model is allowed to produce. |
| `batch decode` | Turning generated token IDs back into normal readable text. |
| `Hugging Face Hub` | The website/service used here to download models and datasets. |
| `LoRA` | A lightweight fine-tuning method that trains a small set of extra weights instead of the full model. |
| `PEFT` | Short for parameter-efficient fine-tuning; methods that adapt a model without retraining everything. |
| `TRL` | A training library for language-model alignment and instruction-style training workflows. |
| `BLEU` | A metric that compares machine-generated text to reference text for caption quality. |
| `ROUGE` | A metric that measures overlap between generated text and reference text. |
| `CUDA` | NVIDIA's system for running GPU computations. |
| `CPU` | The main processor in the computer; slower than the GPU for big model inference. |
| `RGB` | A common image format with red, green, and blue color channels. |
| `PNG` | A lossless image format often used for synthetic or clean images. |
| `JPG` | A compressed image format commonly used for photos and quick sample images. |
| `zip archive` | A compressed file that bundles multiple files together. |
| `JSON annotation` | A text file that stores structured labels or metadata for each sample. |
| `dataset` | A collection of data examples used for training or evaluation. |
| `smoke test` | A very small test that checks whether the pipeline basically works. |

