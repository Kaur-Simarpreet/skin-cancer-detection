# 🔬 Multimodal Skin Cancer Detection — ISIC 2024

A deep learning system for automated skin cancer detection using the **ISIC 2024 dataset**, combining dermoscopic image analysis with structured clinical metadata across three multimodal fusion strategies.

> **Course Project** — AI for Biomedical Research  
> **Best Result:** Late Fusion achieves **AUC = 0.9189** on the holdout test set

---

## 👥 Contributors

| **Akash Kadam** |
| **Simarpreet Kaur** | 
| **M. Bhagya Sri** | 
| **Harshini P.** | 

---

## 📁 Repository Structure

```
multimodal-skin-cancer-detection/
├── Multimodal_Skin_Cancer_Detection.ipynb   # Main notebook (all phases)
├── requirements.txt                          # Python dependencies
├── .gitignore                                # Excludes data & model weights
└── README.md
```

---

## 🗃️ Dataset

This project uses the **ISIC 2024 — Skin Cancer Detection with 3D-TBP** dataset.

📂 **Data Folder (Google Drive):**  
👉 https://drive.google.com/drive/folders/198-NheTFWW1A2UkalhHDYfyYGPKY6Ppd

The folder contains:
- `train-metadata.csv` — Clinical features and labels (~400k records)
- `train-image.hdf5` — Dermoscopic images keyed by `isic_id`

> The notebook automatically downloads the data from the Drive link above. **No manual setup needed.**

---

## 🚀 How to Run

### ✅ Recommended: Google Colab (No Setup Required)

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Set the runtime to **GPU** → `Runtime > Change runtime type > T4 GPU`
3. Run **Cell 0** first — it auto-mounts Drive and downloads the data
4. Run all remaining cells in order

### 💻 Local Machine

1. Clone the repo:
   ```bash
   git clone https://github.com/Kaur-Simarpreet/multimodal-skin-cancer-detection.git
   cd multimodal-skin-cancer-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `data/` folder and place the dataset files inside:
   ```
   data/
   ├── train-metadata.csv
   └── train-image.hdf5
   ```

4. Open the notebook in Jupyter and run all cells.

---

## 🧠 Methodology

### Phase 1 — Data Engineering
- Patient-level **70 / 15 / 15** stratified split (prevents data leakage)
- Clinical feature engineering: `age_approx`, `sex`, `tbp_lv_areaMM2`, `tbp_lv_perimeterMM`, `tbp_lv_H`, `tbp_lv_L`
- Benign class downsampling for training efficiency

### Phase 2 — Vision Baselines (Unimodal)

| Model | AUC |
|---|---|
| ResNet50 | ~0.87 |
| EfficientNet-B0 | ~0.88 |
| **EfficientNet-B3** | **0.8999** ✅ |
| ViT-B/16 | ~0.87 |

### Phase 3 — Clinical Baselines (Tabular)

| Model | AUC |
|---|---|
| XGBoost (Exp 7) | **0.8695** ✅ |

### Phase 5 — Multimodal Fusion

| Strategy | Description | AUC |
|---|---|---|
| **Early Fusion** | Metadata tiled as extra image channels (9-channel EfficientNet) | ~0.88 |
| **Intermediate Fusion** | CNN + MLP features concatenated before classification | ~0.89 |
| **Late Fusion** 🏆 | Weighted average: 56% CNN + 44% XGBoost | **0.9189** |

### Phase 6 — Explainability & Robustness
- **SHAP** — clinical feature importance analysis
- **Grad-CAM** — visual saliency maps on dermoscopic images
- **DeLong's Test** — statistical significance of AUC improvement (p < 0.05)
- **Demographic Fairness** — performance across age and sex cohorts
- **Fault Tolerance** — robustness under simulated missing metadata

---

## 📊 Key Results

| Model | AUC | Notes |
|---|---|---|
| Vision Only (EfficientNet-B3) | 0.8999 | Best unimodal vision model |
| Clinical Only (XGBoost) | 0.8695 | Best tabular model |
| Early Fusion | ~0.88 | Input-level fusion |
| Intermediate Fusion | ~0.89 | Feature-level fusion |
| **Late Fusion** | **0.9189** | 🏆 Best overall — weighted ensemble |

The multimodal Late Fusion model outperforms both unimodal baselines by a statistically significant margin (DeLong's test, p < 0.05), demonstrating the clinical value of combining image and metadata modalities.

---

## 📦 Dependencies

Key libraries used:

- `torch`, `torchvision` — deep learning (EfficientNet-B3, ViT)
- `xgboost` — tabular model
- `h5py` — reading HDF5 image store
- `scikit-learn` — preprocessing, metrics
- `shap` — explainability
- `grad-cam` — visual saliency
- `pandas`, `numpy`, `matplotlib`, `seaborn` — data processing & visualization

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is for **academic and research purposes only** as part of the AI for Biomedical Research course.  
The ISIC 2024 dataset is subject to its own [terms of use](https://challenge.isic-archive.com/terms-of-use/).
