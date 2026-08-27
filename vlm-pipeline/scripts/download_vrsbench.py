"""
download_vrsbench.py
--------------------
Downloads VRSBench dataset files from HuggingFace (xiang709/VRSBench)
and extracts validation/sample subsets for GeoNLI captioning, VQA, and grounding.

Usage:
    python scripts/download_vrsbench.py --eval-only      # Downloads annotations + validation images (~3.8 GB)
    python scripts/download_vrsbench.py --annotations    # Downloads annotations JSONs only (~100 MB)
    python scripts/download_vrsbench.py --full           # Downloads full training + val dataset (~12 GB)
"""

import argparse
import json
import os
import sys
import zipfile
from huggingface_hub import hf_hub_download

REPO_ID = "xiang709/VRSBench"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def download_file(filename: str, target_dir: str) -> str:
    print(f"→ Downloading {filename} ...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=target_dir,
    )
    print(f"  ✅ Saved: {local_path}")
    return local_path

def extract_zip(zip_path: str, extract_to: str):
    print(f"→ Extracting {os.path.basename(zip_path)} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"  ✅ Extracted.")

def main():
    parser = argparse.ArgumentParser(description="Download VRSBench dataset for GeoNLI")
    parser.add_argument("--mode", type=str, choices=["annotations", "eval", "full"], default="annotations",
                        help="Download mode: annotations (~100MB), eval (~3.8GB), or full (~12GB)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "vrsbench"),
                        help="Target output directory for VRSBench data")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  VRSBench DATASET DOWNLOADER (ISRO GeoNLI)")
    print("=" * 60)
    print(f"  Repository: {REPO_ID}")
    print(f"  Target Dir: {output_dir}")
    print(f"  Mode      : {args.mode}")
    print()

    # 1. Always download key JSON annotation files
    annotation_files = [
        "VRSBench_EVAL_Cap.json",
        "VRSBench_EVAL_vqa.json",
        "VRSBench_EVAL_referring.json",
        "VRSBench_train.json",
    ]

    for fname in annotation_files:
        try:
            download_file(fname, output_dir)
        except Exception as e:
            print(f"❌ Failed to download {fname}: {e}")

    # 2. Annotations zip
    try:
        val_ann_path = download_file("Annotations_val.zip", output_dir)
        extract_zip(val_ann_path, os.path.join(output_dir, "annotations_val"))
    except Exception as e:
        print(f"⚠️ Could not download Annotations_val.zip: {e}")

    # 3. If eval or full, download images
    if args.mode in ["eval", "full"]:
        try:
            print("\nDownloading Images_val.zip (~3.8 GB) ...")
            val_img_path = download_file("Images_val.zip", output_dir)
            extract_zip(val_img_path, os.path.join(output_dir, "images_val"))
        except Exception as e:
            print(f"❌ Failed to download Images_val.zip: {e}")

    if args.mode == "full":
        try:
            print("\nDownloading Images_train.zip (~8 GB) ...")
            train_img_path = download_file("Images_train.zip", output_dir)
            extract_zip(train_img_path, os.path.join(output_dir, "images_train"))
            train_ann_path = download_file("Annotations_train.zip", output_dir)
            extract_zip(train_ann_path, os.path.join(output_dir, "annotations_train"))
        except Exception as e:
            print(f"❌ Failed to download full training dataset: {e}")

    # Inspect downloaded annotations
    inspect_annotations(output_dir)

def inspect_annotations(output_dir: str):
    print("\n" + "=" * 60)
    print("  VRSBench ANNOTATIONS SUMMARY")
    print("=" * 60)

    eval_cap_path = os.path.join(output_dir, "VRSBench_EVAL_Cap.json")
    eval_vqa_path = os.path.join(output_dir, "VRSBench_EVAL_vqa.json")
    train_path    = os.path.join(output_dir, "VRSBench_train.json")

    if os.path.exists(eval_cap_path):
        with open(eval_cap_path, "r", encoding="utf-8") as f:
            cap_data = json.load(f)
            print(f"  [Captioning Eval Set] : {len(cap_data)} items loaded")
            if cap_data:
                first = cap_data[0]
                print(f"    Sample Image ID : {first.get('image_id', first.get('id', 'N/A'))}")
                print(f"    Sample Caption  : {str(first.get('caption', first.get('conversations', 'N/A')))[:120]}...")

    if os.path.exists(eval_vqa_path):
        with open(eval_vqa_path, "r", encoding="utf-8") as f:
            vqa_data = json.load(f)
            print(f"  [VQA Eval Set]        : {len(vqa_data)} items loaded")

    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
            print(f"  [Train Set]           : {len(train_data)} items loaded")

    print("=" * 60)

if __name__ == "__main__":
    main()
