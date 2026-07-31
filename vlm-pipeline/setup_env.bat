@echo off
REM ============================================================
REM  setup_env.bat
REM  Creates the local environment and installs dependencies.
REM ============================================================

echo.
echo ============================================================
echo   GeoNLI VLM Pipeline - Environment Setup
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo [OK] Virtual environment activated.
echo.

python -m pip install --upgrade pip -q

echo [1/4] Installing PyTorch 2.4.1 with CUDA 12.1 support...
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 ^
    --index-url https://download.pytorch.org/whl/cu121 -q
echo       Done.
echo.

echo [2/4] Installing transformers ecosystem and Qwen2-VL dependencies...
pip install ^
    transformers>=4.45.0 ^
    accelerate>=0.34.0 ^
    huggingface-hub>=0.24.0 ^
    tokenizers>=0.19.0 ^
    safetensors>=0.4.0 ^
    qwen-vl-utils>=0.0.8 ^
    einops>=0.7.0 ^
    timm>=1.0.0 -q
echo       Done.
echo.

echo [3/4] Installing quantization support...
pip install bitsandbytes>=0.43.0 -q
echo       Done.
echo.

echo [4/4] Installing image, LoRA, and evaluation packages...
pip install ^
    Pillow>=10.0.0 ^
    numpy>=1.24.0 ^
    opencv-python>=4.8.0 ^
    peft>=0.12.0 ^
    trl>=0.11.0 ^
    datasets>=2.20.0 ^
    sentencepiece>=0.2.0 ^
    nltk>=3.8.0 ^
    rouge-score>=0.1.2 ^
    evaluate>=0.4.0 ^
    tqdm>=4.66.0 ^
    pdfplumber -q
echo       Done.
echo.

echo ============================================================
echo   Verifying CUDA availability...
echo ============================================================
python -c "import torch; print('[PyTorch]', torch.__version__); print('[CUDA available]', torch.cuda.is_available()); print('[CUDA version]', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('[GPU]', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
echo.

echo ============================================================
echo   Setup complete!
echo.
echo   Next steps:
echo     1. python download_model.py
echo     2. python test_inference.py
echo ============================================================
pause
