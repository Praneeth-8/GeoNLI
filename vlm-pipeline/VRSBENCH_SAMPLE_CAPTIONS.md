# VRSBench Sample Photo Captions

This file records a caption for each locally available VRSBench sample image in `data/vrsbench/images_sample/Images_val/`.

## Caption List

| Image | Caption |
| --- | --- |
| `07160_0000.png` | An aerial view of an industrial site with a large circular storage tank or basin, surrounding buildings, and nearby paved ground. |
| `07306_0000.png` | A sports or school complex with an oval running track, a central field, nearby roads, and several buildings around the edge. |
| `09179_0000.png` | A dry agricultural area with a large field, visible track marks, and a building at the lower edge of the scene. |
| `10904_0000.png` | A roadside scene with a parking area, a narrow green median, and a vehicle traveling along a divided road. |
| `P0837_0000.png` | A small marina with several boats docked along a pier beside dark water. |
| `P0998_0016.png` | A waterfront residential area with multiple docks extending into the water from houses along the shoreline. |
| `P1022_0015.png` | A grayscale aerial view showing a river or canal, roads, nearby structures, and dense tree cover. |
| `P1023_0001.png` | A dense urban district with intersecting roads, tightly packed buildings, and mixed residential and commercial blocks. |
| `P1647_0013.png` | A low-contrast aerial scene with a waterway, a crossing road or bridge, scattered buildings, and surrounding vegetation. |
| `P2714_0044.png` | A large urban area with a main arterial road, compact city blocks, and a mix of industrial and residential structures. |

## Testing Status

- `python -m py_compile vlm-pipeline/test_inference.py vlm-pipeline/download_model.py` passed.
- The sample images open successfully with Pillow.
- A real inference smoke test on `P0837_0000.png` could not complete in this environment because `torch` is not installed.

## Next Step

Install the Python dependencies from `vlm-pipeline/requirements.txt` or run `setup_env.bat`, then rerun `python vlm-pipeline/test_inference.py --image <sample>` for full model-based captioning.
