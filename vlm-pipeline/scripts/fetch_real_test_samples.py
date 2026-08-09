"""
fetch_real_test_samples.py
--------------------------
Downloads sample high-resolution satellite imagery tiles directly
for quick local testing with test_inference.py
"""

import os
import sys
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Sample high-res satellite image URLs (optical satellite imagery)
SAMPLE_IMAGES = {
    "airport.jpg": "https://raw.githubusercontent.com/open-mmlab/mmrotate/main/demo/demo.jpg",
    "port_harbor.jpg": "https://images.unsplash.com/photo-1508873696983-2df515122519?w=1024&q=80",
    "urban_city.jpg": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=1024&q=80",
    "agricultural_fields.jpg": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1024&q=80",
}

def main():
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "test_samples"))
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 60)
    print("  FETCHING REAL SATELLITE TEST IMAGES")
    print("=" * 60)

    for fname, url in SAMPLE_IMAGES.items():
        out_path = os.path.join(target_dir, fname)
        print(f"→ Fetching {fname} ...")
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                print(f"  ✅ Saved: {out_path}  ({len(resp.content)//1024} KB)")
            else:
                print(f"  ❌ Failed (HTTP {resp.status_code})")
        except Exception as e:
            print(f"  ❌ Error downloading {fname}: {e}")

    print("=" * 60)
    print(f"  All test samples saved to: {target_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
