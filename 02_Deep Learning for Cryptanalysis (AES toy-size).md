# Capstone Project — Deep Learning for AES Cryptanalysis

**Môn:** NT219 - Cryptography

**Tiêu đề đề tài:** Deep Learning for AES Cryptanalysis — Sử dụng ML/DL để tấn công và phân tích mật mã (ciphertext‑only, known‑plaintext, chosen‑plaintext, side‑channel)

---

## 1. Tóm tắt đề tài (Project Synopsis)

Đề tài nghiên cứu ứng dụng các kỹ thuật Deep Learning (CNN, RNN, Transformer, GNN, generative models) cho các bài toán **cryptanalysis** thực tế: từ tấn công dựa trên ciphertext/plaintext đến tấn công kênh kề (SCA — power/EM/timing), fault‑assisted attacks, và phân tích mô hình mật mã. Mục tiêu: thiết kế pipeline dữ liệu, xây mô hình DL phù hợp, đánh giá hiệu quả so với các kỹ thuật cổ điển (differential/linear/CPA/DPA), phân tích robustnes/transferability và đề xuất biện pháp phòng thủ.

---

## 2. Mục tiêu học thuật & kỹ năng (Learning Objectives)

1. Nắm các *attack models* trong cryptanalysis: ciphertext‑only, known‑plaintext, chosen‑plaintext, profiling SCA, non‑profiling SCA.
2. Biết xây pipeline từ tạo dataset (synth/real), preprocessing, model design, tới training/validation và thống kê kết quả (guessing entropy, key rank, success rate).
3. Triển khai PoC trên dữ liệu thật (ChipWhisperer / oscilloscope traces) và trên dữ liệu tổng hợp (cipher implementations).
4. Hiểu rủi ro đạo đức/luật pháp, và học cách triển khai thí nghiệm có trách nhiệm (isolated lab, synthetic data, disclosure policy).

---

## 3. Relevance & Motivation

* Các mô hình DL đã cho thấy khả năng khai thác patterns phức tạp mà các phương pháp thống kê cổ điển khó bắt được, cả trên ciphertext‑only tasks (ví dụ phân lớp ciphertext theo khóa hoặc plaintext) và SCA (recover key bytes từ trace).
* Sinh viên sẽ học cách kết hợp kiến thức mật mã và ML — một kỹ năng đang rất cần trong an ninh thông tin hiện đại.

---

## 4. Câu hỏi nghiên cứu & giả thuyết (RQ & Hypotheses)

**RQ1:** DL có thể cải thiện hiệu năng (giảm số trace/queries cần thiết) so với CPA/DPA/traditional attacks trong tấn công SCA cho các cài đặt thực tế (IoT, HSM, TLS) không?

**RQ2:** Các architecture (1D‑CNN vs Transformer vs RNN) và chế độ huấn luyện (supervised profiling vs self‑supervised transfer) ảnh hưởng thế nào đến khả năng khôi phục khóa, đặc biệt khi có misalignment/noise?

**Giả thuyết:** 1D‑CNN được thiết kế tốt + data augmentation thường hiệu quả cho SCA; Transformer/attention có lợi khi trace dài & global context cần; transfer learning giảm bộ data cần cho device mới.

---

## 5. Bối cảnh kỹ thuật & nền tảng (Background)

* **Cryptanalysis tasks:**

  * *Ciphertext‑only:* mô hình DL map ciphertext → plaintext / key/hypothesis (dễ experimental với toy ciphers, khó với AES‑128 thực).
  * *Known/Chosen plaintext:* attacker có khả năng chọn plaintext và quan sát ciphertext → dùng DL để học mapping plaintext→ciphertext/ key features.
  * *SCA (profiling):* attacker có labeled traces from profiled device; supervised DL mô hình hoá leakage → recover keys on target device.
  * *SCA (non‑profiling):* use transfer/unsupervised methods, sometimes with synthetic traces.
* **ML building blocks:** 1D convolutional nets (time‑series), residual blocks, attention/transformer, RNN/LSTM for sequential dependencies, autoencoders for denoising, GANs/diffusion for trace augmentation.

---

## 6. Thành phần hệ thống & tài nguyên (System Components & Resources)

### 6.1. Phần cứng & Lab

* **ChipWhisperer** (recommended) or oscilloscope + probes + EM probes for SCA capture.
* Test devices: microcontrollers (e.g., STM32, Atmel), smartcards/secure elements (for advanced PoC), IoT devices (ESP32), HSM emulator (SoftHSM for protocol testing).
* **Compute:** GPU workstation (NVIDIA RTX 30xx/40xx or A100 if available). For larger experiments, cloud GPU (AWS, GCP) with controlled dataset.

### 6.2. Phần mềm & frameworks

* **DL frameworks:** PyTorch (preferred), TensorFlow/Keras.
* **SCA toolchain:** ChipWhisperer API, scikit‑leakage tools, tools for trace alignment/preprocessing.
* **Auxiliary libs:** NumPy, Pandas, Scikit‑learn, SciPy, librosa (for signal ops).
* **Experiment tracking:** Weights & Biases / TensorBoard.

### 6.3. Dữ liệu

* **Synthetic datasets:** implement toy‑cipher/AES/DES encryptors to generate labeled ciphertext/plaintext/key tuples.
* **Real SCA traces:** capture in lab (profiling & attack sets) with varying SNR, alignment, clock jitter, temperature.
* **Public datasets:** ASCAD (AES SCA), CHES datasets; use as baselines.

---

## 7. Deployment contexts & attack modes (Contexts & Modes)

### 7.1. Contexts (ngữ cảnh triển khai thực tế)

* **Embedded / IoT:** remote devices performing cryptographic ops with limited side‑channel protections — good target for device‑specific profiling.
* **Cloud / VM co‑residency:** microarchitectural leakage (cache/timing) captured via co‑resident VMs — attacker may use DL on collected timing/cache traces.
* **TLS / Web servers:** timing attacks or microarchitectural leakage from cryptographic libraries (OpenSSL libs) — needs careful instrumentation.
* **Smartcards & HSMs:** high‑value targets — physical access may enable richer SCA but with stronger countermeasures.

### 7.2. Attack modes (modes)

* **Ciphertext‑only vs Known‑plaintext vs Chosen‑plaintext:** choose depending on target. DL methods are more practical when attacker controls/knows plaintext or can profile device.
* **Profiling SCA (supervised):** attacker trains on labeled traces from device identical to target.
* **Non‑profiling SCA / Transfer attacks:** attacker trains on similar device or synthetic traces — use domain adaptation/transfer learning.
* **Hybrid attacks:** combine DL with classical cryptanalysis (use DL to recover partial key material or leakage priors then feed to algebraic attack).

---

## 8. Lựa chọn mô hình & tham số (Model choices & hyperparameters)

### 8.1. Model families & when to use

* **1D‑CNN (ResNet‑like):** hiệu quả cho trace SCA, low‑latency inference. Use for point‑of‑interest learning.
* **Transformer / Attention:** when long‑range dependencies matter (e.g., multi‑round leakage) hoặc khi trace length lớn and global context useful.
* **RNN / LSTM:** archaic for SCA, sometimes helpful for sequence-of‑blocks tasks.
* **Autoencoder / Denoiser:** preprocessing to improve SNR and alignment invariance.
* **Generative (GAN / diffusion):** data augmentation for low‑data devices, simulating noise patterns.
* **Hybrid (ML + algebra):** ML predicts intermediate values / subkey candidates; algebraic techniques combine candidates to full key.

### 8.2. Hyperparameters — practical recommendations

* **Input processing:** normalize traces (z‑score), cropping to window around operation, alignment with cross‑correlation or dynamic time warping.
* **CNN:** kernel sizes 3–51 (use larger for coarse features), 4–10 conv blocks, filters doubling per block (32→64→128), residual connections, batchnorm, dropout 0.1–0.5.
* **Transformer:** embedding dim 128–512, heads 4–16, depth 4–12; positional encoding for time index.
* **Optimizer:** AdamW, lr 1e‑3 → 1e‑4 with cosine annealing; weight decay 1e‑5.
* **Batch size:** 32–512 depending on GPU; larger batch for stable gradients; use gradient accumulation if limited.
* **Epochs:** 50–200 with early stopping on validation; use LR schedule & warmup for transformers.
* **Loss:** CrossEntropy for classification (key byte prediction with 256 classes); focal loss if class imbalance; ensemble outputs to improve rank.
* **Regularization:** data augmentation (noise injection, random cropping, jitter), mixup for traces, label smoothing.

---

## 9. Dataset design & generation (Dataset & preprocessing)

### 9.1. For ciphertext‑only tasks

* **Synthetic generation:** choose cipher (toy AES, reduced‑round AES), generate large corpus with random keys and plaintexts; vary keyspace size for tractability.
* **Labeling:** target could be entire key, subkey bytes, or plaintext; choose problem formulation that DL can learn (e.g., map ciphertext→key byte).

### 9.2. For SCA tasks

* **Profiling set:** capture many traces with known key/known plaintext; include repeats under varying noise, clock jitter and temperature.
* **Attack set:** hold out traces from target device(s).
* **Preprocessing:** filtering (low/high pass), alignment, POI selection (correlation/TVLA), dimensionality reduction (PCA/autoencoder).
* **Augmentation:** additive Gaussian noise, random shift/jitter, amplitude scaling, synthetic trace augmentation via GANs.

---

## 10. Evaluation metrics & success criteria (Metrics)

* **Key Rank / Guessing Entropy (GE):** expected position of true key candidate after scoring — crucial SCA metric.
* **Success Rate (SR\@N):** fraction of experiments where true key in top‑N guesses.
* **Traces to Recovery (TTR):** number of traces needed for key recovery at target success probability.
* **Accuracy / F1:** per‑byte classification accuracy.
* **Area Under Curve (AUC) / ROC:** for binary leakage detection tasks.
* **Robustness metrics:** transfer success: model trained on device A tested on device B.
* **Compute & cost:** GPU hours, inference latency, model size.

---

## 11. Experimental plan & PoC suggestions

1. **Baseline experiments:** reproduce published CNN SCA results on public ASCAD dataset.
2. **Ciphertext‑only PoC (toy‑AES):** train models to predict subkey or plaintext nibble for reduced‑round AES; measure success vs brute‑force baseline.
3. **Profiling SCA on hardware:** collect traces on microcontroller for AES; train CNN to predict SBox output; report GE & TTR.
4. **Transfer / domain adaptation:** train on one device, test on another; use fine‑tuning or domain adversarial nets to improve transfer.
5. **Generative augmentation:** use GAN/diffusion to expand dataset and measure improvement in low‑data regime.
6. **Hybrid pipeline:** use DL to recover key candidates for subkey, then combine via AES key schedule algebraic solving to full key.

---

## 12. Reproducibility & engineering practices

* **Environment:** record OS, Python, PyTorch versions; containerize with Docker; provide trained checkpoints & seed.
* **Experiment tracking:** use W\&B or TensorBoard; log hyperparameters and random seeds.
* **Data management:** store datasets with metadata (SNR, alignment, device id), share scripts to regenerate synthetic data.
* **Ethics & safety:** use isolated lab network, synthetic data for ciphertext tasks, responsible disclosure process for vulnerabilities discovered in third‑party libraries.

---

## 13. Defenses, mitigations & red teaming (Mitigations)

### 13.1. Implementation defenses

* **Masking & higher‑order masking:** randomize intermediate values; requires careful evaluation against DL attacks.
* **Constant‑time & constant‑flow implementations:** remove secret‑dependent control flow, table lookups.
* **Shuffling, desynchronization:** random delays or random instruction insertion to reduce alignment.

### 13.2. ML‑centred defenses

* **Adversarial training:** augment training with adversarially perturbed traces to reduce model sensitivity.
* **Ensemble & detection:** use ensembles and detectors to flag suspicious query patterns and abnormal access.
* **Query budgets & rate limiting:** operationally limit attacker ability to collect traces/queries.

---

## 14. Risks, Limitations & Ethical considerations

* **Dual‑use:** results can be used for improving attacks; keep experiments in isolated lab and follow responsible disclosure.
* **Data privacy & legality:** do not target third‑party production devices without permission.
* **Model overfitting & false confidence:** high accuracy on profiling device does not imply real‑world success on other devices.

---

## 15. Timeline & Milestones (12 tuần)

* **Tuần 1–2:** Survey literature (DPA/CPA, Deep SCA papers), setup environment (ChipWhisperer, GPU), select datasets.
* **Tuần 3–4:** Reproduce baseline ASCAD CNN results; implement synthetic cipher generator (toy‑AES).
* **Tuần 5–6:** Train DL models for ciphertext‑only toy experiments; run ablation on model families.
* **Tuần 7–8:** Capture profiling traces on hardware; train CNN & Transformer models; measure GE & TTR.
* **Tuần 9:** Transfer learning experiments & generative augmentation study.
* **Tuần 10:** Hybrid pipeline (DL + algebraic) to recover full key; evaluate robustness.
* **Tuần 11:** Implement/verify defenses (masking, desync) and re‑test models.
* **Tuần 12:** Final report, code repo, reproducible scripts, demo video & presentation.

---

## 16. Deliverables

1. **Mid‑term report:** lit review, environment, baseline reproduction (ASCAD).
2. **Final report:** PoC results, experiments, comparisons with classical attacks, mitigation analysis.
3. **Code repo:** data generation & capture scripts, model code, training checkpoints, evaluation notebooks.
4. **Artifacts:** sample traces (sanitized), trained models, plots (GE vs traces), demo video.

---

## 17. Assessment & Rubric (gợi ý)

* Scientific depth & literature grounding: 25%
* Technical implementation & reproducibility: 30%
* Experimental rigor & analysis (metrics & baselines): 30%
* Report quality & ethical handling: 15%

---

## 18. Tools & resources gợi ý

* **Hardware:** ChipWhisperer, oscilloscope + probes, STM32/ESP32 dev boards, smartcards (option).
* **DL libs:** PyTorch, PyTorch Lightning, TensorFlow/Keras.
* **SCA datasets:** ASCAD, CHES challenge datasets.
* **Utilities:** NumPy, Pandas, SciPy, scikit‑learn, librosa, WandB/TensorBoard, Docker.

---

## 19. Extensions & Future Work

* **Unsupervised / self‑supervised attacks:** study contrastive learning to reduce profiling needs.
* **Adversarial ML + certifiable defenses:** formal robustness metrics for leakage models.
* **Cross‑cipher transfer:** can models trained on toy ciphers generalize to real ciphers or to PQC schemes?

---

## 20. Appendix: Repository Structure (mẫu)

```
project-root/
  ├─ data/               # synthetic generator, ASCAD import scripts, metadata
  ├─ capture/            # ChipWhisperer capture scripts, oscilloscope configs
  ├─ models/             # CNN/Transformer architectures + training scripts
  ├─ experiments/        # notebooks, hyperparam sweeps, evaluation scripts
  ├─ infra/              # Dockerfiles, requirements.txt, GPU run scripts
  ├─ docs/               # report, slides, ethics note
  └─ artifacts/          # trained checkpoints, sanitized sample traces
```

---
