# Deep Learning for AES Cryptanalysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

> Nghien cuu ung dung Deep Learning (CNN, Transformer) de tan cong va phan tich mat ma AES thong qua cac kich ban: ciphertext-only, known-plaintext, chosen-plaintext va side-channel analysis.

---

## Thong tin do an

| | |
|---|---|
| **Mon hoc** | NT219 — Cryptography |
| **De tai** | Deep Learning for AES Cryptanalysis |
| **Truong** | Truong Dai hoc Cong nghe Thong tin — DHQG TP.HCM |

### Thanh vien nhom

| Ho va ten | MSSV |
|---|---|
| Nguyen Tan Danh | 24520262 |
| Nguyen Thi Tuyet Nhi | 24521263 |
| Nguyen Quoc Truong | 24521896 |

---

## Muc luc

- [Tong quan](#tong-quan)
- [Yeu cau he thong](#yeu-cau-he-thong)
- [Cai dat](#cai-dat)
- [Cach su dung](#cach-su-dung)
- [Cau truc Project](#cau-truc-project)
- [Models](#models)
- [Metrics](#metrics)
- [Ket qua](#ket-qua)
- [Chay Test](#chay-test)
- [Dong gop](#dong-gop)
- [License](#license)

---

## Tong quan

Du an thuc hien 4 kich ban tan cong mat ma AES bang Deep Learning:

| # | Thi nghiem | Mo ta | Script |
|---|---|---|---|
| 1 | **Ciphertext-only** | Du doan key byte chi tu ciphertext (toy-AES, reduced rounds) | `01_toy_aes_ciphertext_only.py` |
| 2 | **Known-plaintext** | Du doan key byte tu cap (plaintext, ciphertext) | `02_known_plaintext_attack.py` |
| 3 | **Simulated SCA** | Khoi phuc key tu power traces mo phong (Hamming Weight model) | `03_simulated_sca.py` |
| 4 | **Chosen-plaintext** | Phan tich differential voi plaintext duoc chon | `04_chosen_plaintext_attack.py` |

Ngoai ra, du an cung cai dat cac tan cong co dien (CPA, DPA) de so sanh hieu qua voi phuong phap DL.

---

## Yeu cau he thong

- **Python** >= 3.9
- **NVIDIA GPU** voi CUDA support (khuyen nghi, khong bat buoc)
- **pip** (Python package manager)

---

## Cai dat

### 1. Clone repository

```bash
git clone https://github.com/ntd4nh/mmh_NT219.Q22.ANTT.git
cd mmh_NT219.Q22.ANTT
```

### 2. Tao virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Cai PyTorch

```bash
# Voi GPU (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Chi CPU (khong co GPU)
pip install torch torchvision torchaudio
```

> Kiem tra phien ban CUDA tuong thich tai [pytorch.org/get-started](https://pytorch.org/get-started/locally/)

### 4. Cai cac dependencies con lai

```bash
pip install -r requirements.txt
```

### 5. Kiem tra cai dat

```bash
# Kiem tra GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Kiem tra AES implementation
python utils/aes_ops.py
# Expected: All AES verification tests passed!
```

---

## Cach su dung

### Experiment 1: Ciphertext-only Attack

```bash
# Co ban
python experiments/01_toy_aes_ciphertext_only.py --rounds 2 --epochs 50

# Voi data augmentation
python experiments/01_toy_aes_ciphertext_only.py --rounds 2 --epochs 50 --augment

# Nhanh (it samples)
python experiments/01_toy_aes_ciphertext_only.py --rounds 2 --epochs 10 --train-samples 50000
```

### Experiment 2: Known-Plaintext Attack

```bash
# Mot round count
python experiments/02_known_plaintext_attack.py --rounds 4 --epochs 50

# Sweep nhieu rounds (1-6)
python experiments/02_known_plaintext_attack.py --sweep-rounds --epochs 40
```

### Experiment 3: Simulated SCA

```bash
# Co ban
python experiments/03_simulated_sca.py --snr 5.0 --epochs 50

# Day du pipeline: augmentation + autoencoder + so sanh CPA
python experiments/03_simulated_sca.py --snr 5.0 --epochs 50 --augment --use-autoencoder --compare-cpa
```

### Experiment 4: Chosen-Plaintext Attack

```bash
# Differential analysis
python experiments/04_chosen_plaintext_attack.py --rounds 3 --epochs 50

# Sweep rounds
python experiments/04_chosen_plaintext_attack.py --sweep-rounds --epochs 40
```

> **Tip:** Them `--model cnn` hoac `--model transformer` de chi train mot model.

---

## Cau truc Project

```
Project/
├── data/synthetic/              # Dataset generators
│   └── generator.py             # 4 loai dataset cho 4 attack mode
├── models/                      # Kien truc DL
│   ├── cnn.py                   # SmallCNN, DeepCNN (ResNet-style 1D)
│   ├── transformer.py           # CryptoTransformer, CryptoTransformerSCA
│   └── autoencoder.py           # DenoisingAutoencoder
├── attacks/                     # Cac phuong phap tan cong
│   ├── classical.py             # CPA, DPA (baseline co dien)
│   └── dl_attack.py             # DL-based key recovery
├── evaluation/                  # Danh gia & truc quan hoa
│   ├── metrics.py               # GE, SR@N, TTR, Key Rank
│   └── visualize.py             # Cac ham ve bieu do
├── experiments/                 # Script chay thi nghiem
│   ├── 01_toy_aes_ciphertext_only.py
│   ├── 02_known_plaintext_attack.py
│   ├── 03_simulated_sca.py
│   └── 04_chosen_plaintext_attack.py
├── utils/
│   ├── aes_ops.py               # AES-128 (FIPS 197)
│   └── preprocessing.py         # Normalization, TraceAugmentor
├── artifacts/results/           # Ket qua training (plots, JSON)
├── config.yaml                  # Hyperparameters
└── requirements.txt             # Dependencies
```

---

## Models

| Model | Ung dung | Input |
|-------|----------|-------|
| **SmallCNN** | Phan tich ciphertext/plaintext | 16-48 bytes |
| **DeepCNN** | Phan tich power traces | 700+ time points |
| **CryptoTransformer** | Byte-level attention | 16-48 bytes |
| **CryptoTransformerSCA** | Patch-based trace analysis | 700+ points |
| **DenoisingAutoencoder** | Khu nhieu & trich xuat features | 700+ points |

## Metrics

| Metric | Y nghia |
|--------|---------|
| **Guessing Entropy (GE)** | So lan doan trung binh de tim dung key |
| **Success Rate (SR@N)** | Ty le key dung nam trong top-N du doan |
| **Traces to Recovery (TTR)** | So traces can de GE = 0 |
| **Key Rank** | Vi tri xep hang cua key dung (0 = tot nhat) |

---

## Ket qua

Ket qua duoc luu trong `./artifacts/results/`:

| File | Noi dung |
|------|---------|
| `*_curves.png` | Bieu do training (loss, accuracy) |
| `ge_vs_traces.png` | Guessing Entropy theo so traces |
| `*_key_rank_dist.png` | Phan phoi Key Rank |
| `model_comparison.png` | So sanh cac models |
| `dl_vs_cpa_comparison.png` | So sanh DL vs CPA |
| `results.json` | Tong hop ket qua (JSON) |

---

## Chay Test

```bash
# Kiem tra AES implementation (FIPS 197 test vectors)
python utils/aes_ops.py

# Kiem tra models
python models/cnn.py
python models/transformer.py

# Kiem tra classical attacks
python attacks/classical.py

# Quick test experiment (2 epochs, it samples)
python experiments/01_toy_aes_ciphertext_only.py --epochs 2 --train-samples 5000 --val-samples 1000 --test-samples 1000
```

---

## Dong gop

1. Fork repository
2. Tao branch moi: `git checkout -b feature/ten-feature`
3. Commit thay doi: `git commit -m "Add: mo ta thay doi"`
4. Push branch: `git push origin feature/ten-feature`
5. Tao Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
