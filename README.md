# Deep Learning for AES Cryptanalysis

> **Môn:** NT219 - Cryptography  
> **Đề tài:** Sử dụng ML/DL để tấn công và phân tích mật mã AES  
> **Attack modes:** Ciphertext-only, Known-plaintext, Chosen-plaintext, Side-channel

---

## 📋 Tổng quan

Dự án nghiên cứu ứng dụng Deep Learning (CNN, Transformer) cho các bài toán **cryptanalysis** trên AES:

1. **Ciphertext-only Attack** — Dự đoán key byte chỉ từ ciphertext (toy-AES, reduced rounds)
2. **Known-plaintext Attack** — Dự đoán key byte từ cặp (plaintext, ciphertext)
3. **Chosen-plaintext Attack** — Phân tích differential với plaintext được chọn
4. **Side-channel Analysis** — Khôi phục key từ power traces (simulated)

## 🏗️ Cấu trúc Project

```
Project/
├── data/synthetic/          # Dataset generators
│   └── generator.py         # 4 attack mode generators
├── models/                  # DL architectures
│   ├── cnn.py              # SmallCNN, DeepCNN (ResNet-style)
│   ├── transformer.py      # CryptoTransformer, CryptoTransformerSCA
│   └── autoencoder.py      # DenoisingAutoencoder
├── attacks/                 # Attack implementations
├── evaluation/              # Metrics & visualization
│   ├── metrics.py          # GE, SR, TTR, Key Rank
│   └── visualize.py        # Plot functions
├── experiments/             # Runnable experiment scripts
│   ├── 01_toy_aes_ciphertext_only.py
│   ├── 02_known_plaintext_attack.py
│   └── 03_simulated_sca.py
├── utils/
│   └── aes_ops.py          # AES-128 implementation
├── config.yaml             # Hyperparameters
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify AES implementation

```bash
python utils/aes_ops.py
# Output: ✅ All AES verification tests passed!
```

### 3. Chạy Experiment 1: Ciphertext-only Attack (Toy-AES)

```bash
# 2-round AES, 50 epochs, cả CNN và Transformer
python experiments/01_toy_aes_ciphertext_only.py --rounds 2 --epochs 50

# Chỉ CNN, 1-round AES (nhanh nhất)
python experiments/01_toy_aes_ciphertext_only.py --rounds 1 --model cnn --epochs 30
```

### 4. Chạy Experiment 2: Known-Plaintext Attack

```bash
# Single round count
python experiments/02_known_plaintext_attack.py --rounds 4 --epochs 50

# Sweep qua nhiều rounds (1-6) để xem accuracy giảm
python experiments/02_known_plaintext_attack.py --sweep-rounds --epochs 40
```

### 5. Chạy Experiment 3: Simulated SCA

```bash
python experiments/03_simulated_sca.py --snr 5.0 --epochs 50
```

## 📊 Metrics

| Metric | Ý nghĩa |
|--------|---------|
| **Guessing Entropy (GE)** | Số lần đoán trung bình để tìm đúng key |
| **Success Rate (SR@N)** | Tỷ lệ key đúng nằm trong top-N dự đoán |
| **Traces to Recovery (TTR)** | Số traces cần để GE = 0 |
| **Key Rank** | Vị trí xếp hạng của key đúng |

## 🧠 Models

| Model | Use case | Input |
|-------|----------|-------|
| **SmallCNN** | Ciphertext/plaintext analysis | 16-48 bytes |
| **DeepCNN** | Power trace analysis | 700+ time points |
| **CryptoTransformer** | Byte-level attention | 16-32 bytes |
| **CryptoTransformerSCA** | Patch-based trace | 700+ points |
| **DenoisingAutoencoder** | Trace preprocessing | 700+ points |

## 📁 Output

Kết quả được lưu trong `./artifacts/results/`:
- Training curves (loss, accuracy)
- GE vs traces plots
- Confusion matrices
- Key rank distributions
- JSON summary files
