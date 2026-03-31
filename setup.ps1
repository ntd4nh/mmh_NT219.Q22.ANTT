# ============================================================
# Setup Script for Deep Learning AES Cryptanalysis Project
# ============================================================
# Chạy: .\setup.ps1
# Nếu bị lỗi policy: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Deep Learning for AES Cryptanalysis - Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ---- Tìm Python ----
Write-Host "[1/5] Tim Python..." -ForegroundColor Yellow

$pythonCmd = $null

# Thử các cách tìm Python
$candidates = @("py", "python3", "python")
foreach ($cmd in $candidates) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python \d") {
            $pythonCmd = $cmd
            Write-Host "  Tim thay: $cmd -> $ver" -ForegroundColor Green
            break
        }
    } catch {}
}

# Thử tìm trực tiếp trong AppData
if (-not $pythonCmd) {
    $paths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe",
        "$env:LOCALAPPDATA\Python\PythonCore-*\python.exe",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\python*.exe",
        "C:\Python*\python.exe"
    )
    foreach ($pattern in $paths) {
        $found = Get-ChildItem $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            $pythonCmd = $found.FullName
            $ver = & $pythonCmd --version 2>&1
            Write-Host "  Tim thay: $pythonCmd -> $ver" -ForegroundColor Green
            break
        }
    }
}

if (-not $pythonCmd) {
    Write-Host "  KHONG TIM THAY PYTHON!" -ForegroundColor Red
    Write-Host "  Hay cai dat Python tu https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "  Nho tick 'Add Python to PATH' khi cai dat!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# ---- Cài dependencies ----
Write-Host "[2/5] Cai dat dependencies..." -ForegroundColor Yellow

& $pythonCmd -m pip install --upgrade pip 2>&1 | Out-Null
& $pythonCmd -m pip install numpy torch pandas scikit-learn scipy matplotlib seaborn pyyaml tqdm tensorboard h5py pycryptodome 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "  LOI khi cai dependencies!" -ForegroundColor Red
    Write-Host "  Thu chay thu cong: $pythonCmd -m pip install numpy" -ForegroundColor Yellow
    exit 1
}
Write-Host "  Dependencies da cai xong!" -ForegroundColor Green
Write-Host ""

# ---- Verify imports ----
Write-Host "[3/5] Kiem tra imports..." -ForegroundColor Yellow

$testScript = @"
import sys
print(f'Python: {sys.executable}')
print(f'Version: {sys.version}')
modules = ['numpy', 'torch', 'pandas', 'sklearn', 'scipy', 'matplotlib', 'yaml', 'tqdm', 'h5py', 'Crypto']
ok = True
for m in modules:
    try:
        __import__(m)
        print(f'  [OK] {m}')
    except ImportError:
        print(f'  [FAIL] {m}')
        ok = False
if ok:
    print('\nTat ca modules OK!')
else:
    print('\nMot so modules chua cai duoc!')
    sys.exit(1)
"@

& $pythonCmd -c $testScript

if ($LASTEXITCODE -ne 0) {
    Write-Host "  Mot so modules chua cai duoc!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ---- Verify AES ----
Write-Host "[4/5] Verify AES implementation..." -ForegroundColor Yellow
& $pythonCmd utils\aes_ops.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "  AES verification PASSED!" -ForegroundColor Green
} else {
    Write-Host "  AES verification FAILED!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ---- Test model creation ----
Write-Host "[5/5] Test model creation..." -ForegroundColor Yellow

$modelTest = @"
import sys
sys.path.insert(0, '.')
from models.cnn import SmallCNN, DeepCNN, count_parameters
from models.transformer import CryptoTransformer
import torch

# Test SmallCNN
m1 = SmallCNN(input_size=16, num_classes=256)
x1 = torch.randn(2, 16)
y1 = m1(x1)
p1 = sum(p.numel() for p in m1.parameters())
print(f'  SmallCNN:         {p1:>10,} params | output: {tuple(y1.shape)}')

# Test DeepCNN
m2 = DeepCNN(input_size=700, num_classes=256)
x2 = torch.randn(2, 700)
y2 = m2(x2)
p2 = sum(p.numel() for p in m2.parameters())
print(f'  DeepCNN:          {p2:>10,} params | output: {tuple(y2.shape)}')

# Test Transformer
m3 = CryptoTransformer(input_size=16, num_classes=256)
x3 = torch.randint(0, 256, (2, 16))
y3 = m3(x3.float())
p3 = sum(p.numel() for p in m3.parameters())
print(f'  CryptoTransformer:{p3:>10,} params | output: {tuple(y3.shape)}')

# Check GPU
if torch.cuda.is_available():
    print(f'\n  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('\n  GPU: Khong co (se dung CPU)')

print('\nTat ca models OK!')
"@

& $pythonCmd -c $modelTest

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Models test PASSED!" -ForegroundColor Green
} else {
    Write-Host "  Models test FAILED!" -ForegroundColor Red
    exit 1
}

# ---- Tạo thư mục output ----
New-Item -ItemType Directory -Force -Path "artifacts\results\ciphertext_only" | Out-Null
New-Item -ItemType Directory -Force -Path "artifacts\results\known_plaintext" | Out-Null  
New-Item -ItemType Directory -Force -Path "artifacts\results\sca_simulated" | Out-Null
New-Item -ItemType Directory -Force -Path "artifacts\checkpoints" | Out-Null
New-Item -ItemType Directory -Force -Path "data\generated" | Out-Null

# ---- Done ----
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  SETUP HOAN TAT!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Cac lenh chay experiment:" -ForegroundColor Cyan
Write-Host "  # Experiment 1: Ciphertext-only (nhanh, ~5 phut)" -ForegroundColor White
Write-Host "  $pythonCmd experiments\01_toy_aes_ciphertext_only.py --rounds 1 --model cnn --epochs 30 --train-samples 50000" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Experiment 2: Known-plaintext" -ForegroundColor White
Write-Host "  $pythonCmd experiments\02_known_plaintext_attack.py --rounds 4 --epochs 50" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Experiment 3: Simulated SCA" -ForegroundColor White
Write-Host "  $pythonCmd experiments\03_simulated_sca.py --snr 5.0 --epochs 50" -ForegroundColor Yellow
Write-Host ""
