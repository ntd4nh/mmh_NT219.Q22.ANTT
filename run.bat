@echo off
chcp 65001 >nul 2>&1

REM === Python path đúng ===
set PYTHON=C:\Users\metan\AppData\Local\Python\pythoncore-3.14-64\python.exe

echo.
echo ============================================================
echo   Deep Learning for AES Cryptanalysis - Run Experiments
echo ============================================================
echo.
echo Chon experiment de chay:
echo.
echo   [1] Ciphertext-only Attack (Toy-AES, 1 round, nhanh ~5 phut)
echo   [2] Ciphertext-only Attack (Toy-AES, 2 rounds, day du)
echo   [3] Known-Plaintext Attack (4 rounds)
echo   [4] Known-Plaintext Round Sweep (1-6 rounds, lau)
echo   [5] Simulated SCA (power traces)
echo   [6] Chay tat ca (lau)
echo   [0] Thoat
echo.

set /p choice=Nhap lua chon (0-6): 

if "%choice%"=="1" (
    echo.
    echo === Experiment 1: Ciphertext-only, 1-round AES ===
    "%PYTHON%" experiments\01_toy_aes_ciphertext_only.py --rounds 1 --model cnn --epochs 30 --train-samples 50000
)

if "%choice%"=="2" (
    echo.
    echo === Experiment 1: Ciphertext-only, 2-round AES (CNN + Transformer) ===
    "%PYTHON%" experiments\01_toy_aes_ciphertext_only.py --rounds 2 --epochs 50
)

if "%choice%"=="3" (
    echo.
    echo === Experiment 2: Known-Plaintext, 4-round AES ===
    "%PYTHON%" experiments\02_known_plaintext_attack.py --rounds 4 --epochs 50
)

if "%choice%"=="4" (
    echo.
    echo === Experiment 2: Known-Plaintext Round Sweep (1-6 rounds) ===
    "%PYTHON%" experiments\02_known_plaintext_attack.py --sweep-rounds --epochs 30 --train-samples 50000
)

if "%choice%"=="5" (
    echo.
    echo === Experiment 3: Simulated SCA ===
    "%PYTHON%" experiments\03_simulated_sca.py --snr 5.0 --epochs 50
)

if "%choice%"=="6" (
    echo.
    echo === Chay TAT CA experiments ===
    echo.
    echo --- Experiment 1: Ciphertext-only ---
    "%PYTHON%" experiments\01_toy_aes_ciphertext_only.py --rounds 2 --epochs 50
    echo.
    echo --- Experiment 2: Known-Plaintext ---
    "%PYTHON%" experiments\02_known_plaintext_attack.py --rounds 4 --epochs 50
    echo.
    echo --- Experiment 3: Simulated SCA ---
    "%PYTHON%" experiments\03_simulated_sca.py --snr 5.0 --epochs 50
)

if "%choice%"=="0" (
    exit /b 0
)

echo.
echo === Hoan tat! Ket qua luu trong artifacts\results\ ===
echo.
pause
