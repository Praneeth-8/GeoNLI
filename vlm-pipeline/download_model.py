"""
Download a Qwen2-VL model and processor into the local cache.
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="Download Qwen2-VL model from Hugging Face")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID to download",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="Local directory to cache the model",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 60)
    print("Downloading Qwen2-VL")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Cache: {args.cache_dir}")
    print()

    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError:
        print("transformers is not installed. Run setup_env.bat first.")
        sys.exit(1)

    print("Downloading processor...")
    try:
        AutoProcessor.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"Processor download failed: {exc}")
        sys.exit(1)

    print("Downloading model weights...")
    try:
        Qwen2VLForConditionalGeneration.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"Model download failed: {exc}")
        sys.exit(1)

    print()
    print("Download complete.")
    print("Run: python test_inference.py --image <path>")


if __name__ == "__main__":
    main()
