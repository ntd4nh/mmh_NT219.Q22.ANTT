# Deep Learning cho Phân tích Mật mã AES

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Tình_trạng-Đang_thực_hiện-yellow)

> Nghiên cứu ứng dụng Học Sâu (Deep Learning - CNN, Transformer) để tấn công và phân tích mật mã AES thông qua các kịch bản: chỉ có bản mã (ciphertext-only), biết trước bản rõ (known-plaintext), chọn bản rõ (chosen-plaintext) và phân tích kênh kề (side-channel analysis).

---

## Thông tin đồ án

| | |
|---|---|
| **Môn học** | NT219 — Mật mã học (Cryptography) |
| **Đề tài** | Deep Learning cho Phân tích Mật mã AES |
| **Trường** | Trường Đại học Công nghệ Thông tin — ĐHQG TP.HCM |

## Mục lục

- [Tổng quan](#tổng-quan)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Mô hình xử lý](#mô-hình-xử-lý)
- [Thước đo đánh giá](#thước-đo-đánh-giá)
- [Kết quả](#kết-quả)
- [Chạy kiểm thử](#chạy-kiểm-thử)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)

---

## Tổng quan

Dự án thực hiện 4 kịch bản tấn công mật mã AES bằng Deep Learning:

| # | Thí nghiệm | Mô tả | Tập lệnh |
|---|---|---|---|
| 1 | **Chỉ có bản mã (Ciphertext-only)** | Dự đoán byte khóa chỉ từ bản mã (toy-AES mô phỏng) | `01_toy_aes_ciphertext_only.py` |
| 2 | **Biết trước bản rõ (Known-plaintext)** | Dự đoán byte khóa từ cặp (bản rõ, bản mã) | `02_known_plaintext_attack.py` |
| 3 | **Phân tích kênh kề (Simulated SCA)** | Khôi phục khóa từ các dấu vết năng lượng mô phỏng (mô hình Hamming Weight) | `03_simulated_sca.py` |
| 4 | **Chọn bản rõ (Chosen-plaintext)** | Phân tích vi phân (differential analysis) với bản rõ được thiết kế sẵn | `04_chosen_plaintext_attack.py` |

Ngoài ra, dự án cũng cài đặt các tấn công cổ điển (CPA, DPA) làm nhóm chứng để so sánh hiệu quả với phương pháp Deep Learning.

---

## Yêu cầu hệ thống

- **Python** >= 3.9
- **NVIDIA GPU** có hỗ trợ CUDA (khuyến nghị để rút ngắn thời gian huấn luyện, không bắt buộc)
- **pip** (Trình quản lý gói của Python)

---

## Cài đặt

### 1. Tải mã nguồn về máy (Clone repository)

```bash
git clone https://github.com/ntd4nh/mmh_NT219.Q22.ANTT.git
cd mmh_NT219.Q22.ANTT
```

### 2. Tạo môi trường ảo (Virtual environment)

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Cài đặt PyTorch

```bash
# Phiên bản có GPU (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Phiên bản chỉ dùng CPU (không có GPU)
pip install torch torchvision torchaudio
```

> Kiểm tra phiên bản CUDA tương thích tại [pytorch.org/get-started](https://pytorch.org/get-started/locally/)

### 4. Cài đặt các thư viện phụ thuộc còn lại

```bash
pip install -r requirements.txt
```

### 5. Kiểm tra cài đặt

```bash
# Kiểm tra GPU có nhận diện không
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Kiểm tra thuật toán mã hóa AES thao tác đúng chuẩn chưa
python utils/aes_ops.py
# Kết quả mong đợi: All AES verification tests passed!
```

---

## Cách sử dụng

### Thí nghiệm 1: Phân tích khi chỉ có bản mã (Ciphertext-only)

```bash
# Huấn luyện thông thường (ví dụ: AES 2 vòng, 50 vòng lặp epoch)
python experiments/01_toy_aes_ciphertext_only.py --rounds 2 --epochs 50

# Huấn luyện kèm mở rộng dữ liệu (thêm nhiễu ngẫu nhiên)
python experiments/01_toy_aes_ciphertext_only.py --rounds 2 --epochs 50 --augment

# Lệnh chạy nhanh (ít mẫu, dành cho kiểm thử)
python experiments/01_toy_aes_ciphertext_only.py --rounds 2 --epochs 10 --train-samples 50000
```

### Thí nghiệm 2: Phân tích khi biết bản rõ (Known-Plaintext)

```bash
# Cấu hình cụ thể số vòng AES
python experiments/02_known_plaintext_attack.py --rounds 4 --epochs 50

# Chạy thử nghiệm quét tự động từ 1-6 vòng AES
python experiments/02_known_plaintext_attack.py --sweep-rounds --epochs 40
```

### Thí nghiệm 3: Dấu vết năng lượng mô phỏng (Simulated SCA)

```bash
# Cơ bản
python experiments/03_simulated_sca.py --snr 5.0 --epochs 50

# Đầy đủ quy trình: Mở rộng dữ liệu + Dùng Autoencoder + So sánh tấn công CPA cổ điển
python experiments/03_simulated_sca.py --snr 5.0 --epochs 50 --augment --use-autoencoder --compare-cpa
```

### Thí nghiệm 4: Phân tích khi chọn bản rõ (Chosen-Plaintext)

```bash
# Chạy phân tích vi phân (differential analysis)
python experiments/04_chosen_plaintext_attack.py --rounds 3 --epochs 50

# Quét qua nhiều vòng AES độ phức tạp khác nhau
python experiments/04_chosen_plaintext_attack.py --sweep-rounds --epochs 40
```

> **Gợi ý:** Thêm `--model cnn` hoặc `--model transformer` vào dòng lệnh để chỉ định duy nhất một mô hình dự đoán.

---

## Cấu trúc thư mục

```
Project/
├── data/synthetic/              # Trình tạo bộ dữ liệu nhân tạo
│   └── generator.py             # Sinh 4 loại dữ liệu cho 4 kiểu kịch bản tấn công
├── models/                      # Các kiến trúc học sâu (Deep Learning)
│   ├── cnn.py                   # SmallCNN, DeepCNN (Tương tự ResNet 1D)
│   ├── transformer.py           # CryptoTransformer, CryptoTransformerSCA
│   └── autoencoder.py           # DenoisingAutoencoder
├── attacks/                     # Các thuật toán tấn công (khôi phục khóa)
│   ├── classical.py             # CPA, DPA (Cổ điển làm nhóm chứng)
│   └── dl_attack.py             # Khôi phục khóa bằng sức mạnh học sâu
├── evaluation/                  # Modun đánh giá, đánh giá mô hình
│   ├── metrics.py               # Chứa logic đoán (GE, SR@N, TTR, Key Rank)
│   └── visualize.py             # Hàm vẽ biểu đồ xuất ảnh
├── experiments/                 # Kịch bản dòng lệnh chạy kiểm thử các thí nghiệm
│   ├── 01_toy_aes_ciphertext_only.py
│   ├── 02_known_plaintext_attack.py
│   ├── 03_simulated_sca.py
│   └── 04_chosen_plaintext_attack.py
├── utils/
│   ├── aes_ops.py               # Hiện thực AES-128 chuẩn FIPS 197
│   └── preprocessing.py         # Tiền xử lý dữ liệu, Mở rộng dữ liệu (TraceAugmentor)
├── artifacts/results/           # Lưu trữ phân tích kết quả (Biều đồ ảnh PNG, tệp tin JSON)
├── config.yaml                  # Bộ tham số siêu việt mặc định
└── requirements.txt             # Danh sách thư viện Python
```

---

## Mô hình xử lý

| Mô hình | Ứng dụng | Dữ liệu đầu vào |
|-------|----------|-------|
| **SmallCNN** | Dùng phân tích bản rõ / bản mã | Từ 16 đến 48 bytes |
| **DeepCNN** | Dùng phân tích dấu vết năng lượng rò rỉ | Từ 700+ điểm thời gian |
| **CryptoTransformer** | Chú ý tinh chỉnh mức Byte | Từ 16 đến 48 bytes |
| **CryptoTransformerSCA** | Phân tích dấu vết năng lượng theo mảng/patch | Từ 700+ điểm thời gian |
| **DenoisingAutoencoder** | Tiền xử lý, lọc nhiễu, tách đặc điểm dấu vết kênh kề | Từ 700+ điểm thời gian |

## Thước đo đánh giá

| Thước đo | Ý nghĩa |
|--------|---------|
| **Guessing Entropy (GE)** | Số lần đoán trung bình từ lúc bắt đầu cho đến khi tìm chính xác byte của khóa |
| **Success Rate (SR@N)** | Tỷ lệ khóa chính xác nằm trong nhóm top-N sự lựa chọn cao nhất |
| **Traces to Recovery (TTR)** | Số lượng dấu vết (traces) năng lượng bị rò rỉ tối thiểu cần thu lại để GE tiến về 0 |
| **Key Rank** | Vị trí xếp hạng năng lực dự đoán đúng khóa (Xếp cao nhất = 0 sẽ là hoàn hảo nhất) |

---

## Kết quả

Kết quả sau mỗi lần thí nghiệm sẽ được đóng gói lưu tự động tại `./artifacts/results/`:

| Đặc điểm tệp tin | Nội dung lưu |
|------|---------|
| `*_curves.png` | Biều đồ huấn luyện máy học biểu diễn hàm suy hao (loss) và sai số (accuracy) |
| `ge_vs_traces.png` | Xu hướng Guessing Entropy dựa theo số lượng vết (traces) |
| `*_key_rank_dist.png` | Phân phối thứ hạng dự đoán khóa (Key Rank) |
| `model_comparison.png` | Góc nhìn so sánh sự ưu việt giữa các mô hình học máy |
| `dl_vs_cpa_comparison.png` | Biểu diễn so sánh Tấn công Học sâu (DL) với tấn công Năng lượng (CPA / Cổ điển) |
| `results.json` | Kết xuất tổng hợp thông số thống kê rút ra (đọc dưới dạng văn bản JSON) |

---

## Chạy kiểm thử

```bash
# Kiểm tra sự chính xác trong implementation Thuật toán mã hóa AES chuẩn Hoa Kỳ (FIPS 197 test vectors)
python utils/aes_ops.py

# Kiểm tra kết xuất của các mô hình nơ-ron
python models/cnn.py
python models/transformer.py

# Kiểm tra cơ sở lý thuyết Tấn công Cổ điển
python attacks/classical.py

# Kiểm thử nhanh (chạy lướt sóng qua 2 vòng - ít tham số - ít mẫu phục vụ kiểm chứng hoạt động toàn bộ quy trình không chờ đợi)
python experiments/01_toy_aes_ciphertext_only.py --epochs 2 --train-samples 5000 --val-samples 1000 --test-samples 1000
```

---

## Đóng góp

1. Phân nhánh sao lưu riêng rẽ mã nguồn dự án (Fork the project)
2. Tạo mới một luồng làm việc độc lập: `git checkout -b feature/tuy-bien-gi-do`
3. Thêm chú thích xác nhận thay đổi: `git commit -m "Tính năng: Thêm tính năng gì đó xuất sắc"`
4. Tải luồng mới lên Github: `git push origin feature/tuy-bien-gi-do`
5. Khởi tạo một Góp phần Yêu cầu kéo tự động (Tạo Pull Request)

---

## Giấy phép

Dự án này được phân phối dưới sự chấp thuận của Giấy phép bản quyền MIT. Đọc kỹ file `LICENSE` có sẵn trong source code để phân định rõ mức độ và giới hạn thông tin.
