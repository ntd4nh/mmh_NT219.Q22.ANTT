@echo off
chcp 65001 >nul 2>&1

REM === Python path đúng (có numpy, torch, etc.) ===
set PYTHON=C:\Users\metan\AppData\Local\Python\pythoncore-3.14-64\python.exe

echo.
echo ============================================================
echo   SETUP - Deep Learning for AES Cryptanalysis
echo   Python: %PYTHON%
echo ============================================================
echo.

echo [1/4] Kiem tra Python...
"%PYTHON%" --version
if errorlevel 1 (
    echo LOI: Khong tim thay Python tai %PYTHON%
    pause
    exit /b 1
)
echo.

echo [2/4] Cai dat dependencies...
"%PYTHON%" -m pip install numpy torch pandas scikit-learn scipy matplotlib seaborn pyyaml tqdm tensorboard h5py pycryptodome
echo.

echo [3/4] Verify AES implementation...
"%PYTHON%" utils\aes_ops.py
echo.

echo [4/4] Test models...
"%PYTHON%" -c "import sys; sys.path.insert(0,'.'); from models.cnn import SmallCNN; import torch; m=SmallCNN(16,256); print('SmallCNN OK:', tuple(m(torch.randn(1,16)).shape))"
"%PYTHON%" -c "import sys; sys.path.insert(0,'.'); from models.transformer import CryptoTransformer; import torch; m=CryptoTransformer(16,256); x=torch.randint(0,256,(1,16)); print('Transformer OK:', tuple(m(x.float()).shape))"
"%PYTHON%" -c "import torch; gpu='GPU: '+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'; print(gpu)"
echo.

REM Tao thu muc output
mkdir artifacts\results\ciphertext_only 2>nul
mkdir artifacts\results\known_plaintext 2>nul
mkdir artifacts\results\sca_simulated 2>nul
mkdir data\generated 2>nul

echo ============================================================
echo   SETUP HOAN TAT!
echo ============================================================
echo.
echo Chay experiment bang run.bat
echo.
pause
