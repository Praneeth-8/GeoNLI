# Integrated GeoNLI Test Summary

- Images tested: 5
- Tiler + grounding passed: 5/5
- Captioning passed: 5/5
- VQA/reasoning passed: 5/5
- Total elapsed seconds: 207.11
- CSV results: `integrated_50_image_results.csv`
- JSONL results: `integrated_50_image_results.jsonl`
- Captioned image folder: `captioned_images`

## Sample Outputs

### 1. 05930_0000.png

![Captioned image](captioned_images\001_05930_0000_captioned.png)

- Grounding: pass
- Caption output: The image depicts a serene outdoor setting with a focus on a remote-sensing image of a landscape. The scene features a lush green field with a clear sky overhead. The image captures a section of the field, where a few scattered trees and bushes are visible. The spatial layout suggests a rural or semi-rural area,
- VQA question: What is the shape of the baseball field?
- VQA output: The baseball field in the image appears to be rectangular in shape.
- VQA expected: Diamond

### 2. 06011_0000.png

![Captioned image](captioned_images\002_06011_0000_captioned.png)

- Grounding: pass
- Caption output: The image depicts a remote-sensing image of a landscape with a mix of natural and man-made features. The visible objects include a body of water, possibly a lake or a river, with a small island or landmass in the middle. The water body is surrounded by green vegetation, indicating a lush, possibly tropical or
- VQA question: What color is the vehicle at the top-left corner closest to the top edge?
- VQA output: white
- VQA expected: White

### 3. 06079_0000.png

![Captioned image](captioned_images\003_06079_0000_captioned.png)

- Grounding: pass
- Caption output: The image depicts a remote-sensing image of a landscape with a mix of natural and man-made elements. The visible objects include a colorful sky with patches of blue and white, and a body of water that appears to be a river or lake. The land use includes a field with a yellow crop, possibly corn or wheat
- VQA question: Is the ship occupying the majority of the image frame?
- VQA output: No
- VQA expected: Yes

### 4. 06104_0000.png

![Captioned image](captioned_images\004_06104_0000_captioned.png)

- Grounding: pass
- Caption output: The image depicts a remote-sensing image of a landscape with a mix of natural and man-made features. The terrain is characterized by a variety of vegetation, including trees and grasses, indicating a diverse ecosystem. The spatial layout includes a road system, with visible lane markings and a few vehicles, suggesting human activity. The
- VQA question: What is the main industrial structure visible in the image?
- VQA output: The main industrial structure visible in the image is a factory with a large, multi
- VQA expected: chimney

### 5. 06181_0000.png

![Captioned image](captioned_images\005_06181_0000_captioned.png)

- Grounding: pass
- Caption output: The image depicts a remote-sensing image of a rural landscape. The visible objects include a field with a few scattered trees and a few patches of grass. The spatial layout shows a rectangular field with a clear boundary, indicating agricultural use. The image also captures the natural environment, with a clear sky and a few scattered clouds
- VQA question: Does the windmill have more than two blades?
- VQA output: no
- VQA expected: Yes

