# GeoNLI Setup

## 1. Create Virtual Environment

```bash
cd ~/Project/GeoNLI
python3.12 -m venv venv312
source venv312/bin/activate
pip install --upgrade pip
```

## 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
--index-url https://download.pytorch.org/whl/cu121
```

## 3. Install Dependencies

Comment out PyTorch lines in `requirements.txt`, then:
If not commented
```bash
pip install -r requirements.txt
```

## 4. Verify Installation

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```
## 5. Download the Model

```bash
cd vlm-pipline/
python3 download_model.py
```

This downloads the VLM model weights.his downloads the vlm's weight
## 6. Run

```bash
source venv312/bin/activate
python3 webapp.py
```
