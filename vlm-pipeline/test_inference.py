"""
Run Qwen2-VL inference on a satellite image tile.
"""

import argparse
import os
import sys
import time
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")

DEFAULT_CAPTION_PROMPT = (
    "You are analyzing a satellite or aerial image tile. "
    "Describe what you observe in detail. Include land use types, visible objects, "
    "spatial arrangement, and notable geographic features. Be specific and factual."
)

DEFAULT_VQA_PROMPT = (
    "Look at this satellite image tile. "
    "Question: {question} "
    "Answer concisely and accurately based only on what is visible."
)


def create_placeholder_image(path: str) -> str:
    try:
        import random
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (512, 512), color=(120, 160, 90))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 300, 200], fill=(160, 120, 70))
        draw.ellipse([320, 60, 480, 180], fill=(70, 120, 200))
        for x, y in [(80, 80), (130, 80), (80, 130), (180, 100)]:
            draw.rectangle([x, y, x + 30, y + 25], fill=(210, 210, 210))
        draw.line([(0, 256), (512, 256)], fill=(80, 80, 80), width=8)
        draw.line([(256, 0), (256, 512)], fill=(80, 80, 80), width=6)
        for _ in range(30):
            cx = random.randint(350, 480)
            cy = random.randint(250, 480)
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(40, 120, 40))

        img.save(path)
        return path
    except ImportError:
        print("Pillow is not installed.")
        sys.exit(1)


def build_runtime_config(args) -> dict:
    import torch

    cuda_available = torch.cuda.is_available()
    load_in_8bit = args.load_in_8bit
    load_in_4bit = args.load_in_4bit or (cuda_available and not args.no_quantization and not load_in_8bit)

    return {
        "model_id": args.model,
        "cache_dir": args.cache_dir,
        "cuda_available": cuda_available,
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "no_quantization": args.no_quantization,
    }


def load_model_and_processor(config: dict):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_id = config["model_id"]
    cache_dir = config["cache_dir"]
    cuda_ok = config["cuda_available"]
    load_4bit = config["load_in_4bit"]
    load_8bit = config["load_in_8bit"]
    no_quantization = config["no_quantization"]

    print()
    print(f"Loading model: {model_id}")
    print(f"CUDA available: {cuda_ok}")

    quant_config = None
    dtype = torch.float32
    device_map = "cpu"

    if cuda_ok:
        dtype = torch.float16
        device_map = "auto"
        if no_quantization:
            print("Mode: float16 without quantization")
        else:
            try:
                from transformers import BitsAndBytesConfig

                if load_8bit:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    print("Mode: 8-bit quantization")
                elif load_4bit:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    print("Mode: 4-bit quantization")
            except ImportError:
                print("bitsandbytes is not available. Falling back to standard loading.")
    else:
        print("Mode: CPU float32")

    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"Processor loading failed: {exc}")
        print("Run: python download_model.py")
        sys.exit(1)

    try:
        model_kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": dtype,
            "device_map": device_map,
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        model.eval()
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        sys.exit(1)

    return model, processor


def run_inference(model, processor, image_path: str, prompt: str, max_new_tokens: int = 256) -> str:
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        print(f"Image: {os.path.basename(image_path)} ({width}x{height})")
    except Exception as exc:
        print(f"Could not open image '{image_path}': {exc}")
        sys.exit(1)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    elapsed = time.time() - start

    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    token_count = sum(len(item) for item in generated_ids)
    rate = token_count / elapsed if elapsed else 0
    print(f"Generated {token_count} tokens in {elapsed:.1f}s ({rate:.1f} tok/s)")

    return output_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL satellite image inference")
    parser.add_argument("--image", default=None, help="Path to a PNG or JPG image")
    parser.add_argument("--prompt", default=None, help="Custom prompt")
    parser.add_argument("--vqa", default=None, help="Ask a VQA question instead of captioning")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID to load")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Local model cache")
    parser.add_argument("--load-in-4bit", action="store_true", help="Force 4-bit quantization")
    parser.add_argument("--load-in-8bit", action="store_true", help="Force 8-bit quantization")
    parser.add_argument("--no-quantization", action="store_true", help="Disable quantization")
    args = parser.parse_args()

    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            sys.exit(1)
    else:
        image_path = os.path.join(os.path.dirname(__file__), "data", "sample_tile.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        if not os.path.exists(image_path):
            create_placeholder_image(image_path)

    if args.vqa:
        prompt = DEFAULT_VQA_PROMPT.format(question=args.vqa)
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = DEFAULT_CAPTION_PROMPT

    config = build_runtime_config(args)
    model, processor = load_model_and_processor(config)

    print("Running inference...")
    output = run_inference(model, processor, image_path, prompt, max_new_tokens=args.max_tokens)

    print()
    print("=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
