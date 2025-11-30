# Nudity Detection Algorithm Enhancement  
Pamantasan ng Lungsod ng Maynila — BSCS Thesis (2025)

## 📖 About the Paper
This repository accompanies our thesis:  
**"An Enhancement of Nudity Detection Algorithm for Android Mobile Applications"**  

**Authors:** Abbariao, Jeriel Endrix A. · Ambrocio, Jericho P. · Amper, Miracle Joy F.  
**Advisor:** Prof. Raymund M. Dioses  
**Date:** 2025 (Ongoing)

---

## 📌 Thesis Status
This repository is part of an ongoing undergraduate thesis project at Pamantasan ng Lungsod ng Maynila (BSCS, 2025). The work is currently under development and not yet published. All datasets, models, and results are for academic research purposes only and may change as the thesis progresses.

---

## 🧠 Reference Algorithm

This project builds upon the nudity detection algorithm proposed by **Rahat Yeasin Emon** (2020):  
**"A Novel Nudity Detection Algorithm for Web and Mobile Application Development"**  
Available on [arXiv:2006.01780](https://arxiv.org/abs/2006.01780)

Key components adapted:
- RGB + HSV skin pixel thresholds  
- Face region detection using Google Vision API  
- Ratio-based nudity classification logic

Our enhancements address limitations in Emon's method:
- Bias toward lighter skin tones due to static thresholds  
- Reliance on single-face detection  
- Inability to detect nudity without visible faces  
- Fixed thresholds (0.15, 0.38) replaced with adaptive ML-driven thresholding

---

## 🎯 Objectives
1. Develop an inclusive skin detection algorithm trained on diverse datasets  
2. Detect nudity even when faces are absent, by analyzing body structures  
3. Support multi-face detection (up to 5 faces per image)  
4. Replace static thresholds with adaptive, ML-driven mechanisms  

---

## ⚙️ Prerequisites
- Python 3.8+
- pip
- Virtualenv (recommended)
- CUDA (optional, for GPU acceleration)

---

## 🔧 Environment Configuration (.env)

This project uses a `.env` file to manage paths consistently across scripts.  
The `.env` file is **not included in the repository** (ignored via `.gitignore`).  
You must create it manually in the project root.

### Example `.env`
```env
BASE_DIR=C:/Users/yourname/Nudity-Detection-Android-Train-Model
DATASETS_DIR=${BASE_DIR}/datasets
OUTPUT_DIR=${BASE_DIR}/balanced_output
MODELS_DIR=${BASE_DIR}/models
LOGS_DIR=${BASE_DIR}/logs
```

---

## 📊 Datasets
We used **4 major objectives** with a total of **76,587 images** across 8 datasets.  
Each dataset was carefully **balanced into train/val/test splits** to ensure fairness, inclusivity, and reproducibility.

### Dataset Summary
| Objective   | Train | Val  | Test | Total  |
|-------------|-------|------|------|--------|
| Objective_1 | 8,454 | 1,810| 1,814| 12,078 |
| Objective_2 | 14,700| 3,150| 3,150| 21,000 |
| Objective_3 | 13,173| 2,822| 2,824| 18,819 |
| Objective_4 | 10,451| 7,121| 7,116| 24,688 |
| **Grand Total** | — | — | — | **76,587** |

**Datasets used per objective:**
- **Objective 1:** Skin Tone Dataset, Pratheepan Face Dataset  
- **Objective 2:** COCO 2017, MPII Human Pose Dataset  
- **Objective 3:** WIDER Face Dataset, CelebA Dataset  
- **Objective 4:** P2DatasetFull (Adult Content Dataset), Detector Auto-Generated Dataset  

---

## 📈 Results
We compared our ML models against the baseline RGB/HSV thresholds (Emon's algorithm).  
The ML models consistently outperformed the baseline, especially in precision and F1 scores.

### Final Evaluation Metrics
| Objective   | ML Acc | ML Prec | ML Rec | ML F1 | Baseline Acc | Baseline Prec | Baseline Rec | Baseline F1 |
|-------------|--------|---------|--------|-------|--------------|---------------|--------------|-------------|
| Objective_1 | 0.999  | 1.000   | 0.964  | 0.981 | 0.770        | 0.515         | 0.848        | 0.464       |
| Objective_2 | 0.678  | 0.659   | 0.639  | 0.646 | 0.299        | 0.249         | 0.330        | 0.192       |
| Objective_3 | 0.999  | 0.998   | 0.999  | 0.999 | 0.266        | 0.286         | 0.380        | 0.242       |
| Objective_4 | 0.922  | 0.810   | 0.904  | 0.847 | 0.690        | 0.532         | 0.559        | 0.522       |

---

## 📑 Reviewer Notes
- **Balanced datasets:** Transparent counts logged in `validation_summary.csv` and `validation_totals.csv`.  
- **Confusion matrices:** Exported per objective (e.g., `Objective_1_confusion.csv`).  
- **Training logs:** Detailed logs per objective (e.g., `Objective_1_training_log.csv`) showing epoch-by-epoch convergence.  
- **Reproducibility:** Modular scripts, `.env` path management, reviewer-proof CSVs and models.  
- **Model artifacts:** `.pth` files are the trained PyTorch checkpoints used for ML evaluation. Baseline results come from Emon's threshold algorithm.

---

## ⚠️ Limitations
- Detects nudity only in still images of real humans (JPEG/PNG).  
- Cannot analyze videos, cartoons, animals, or artwork.  
- Supports detection of up to 5 people per image.  
- Accuracy depends on diversity of training datasets; some cultural contexts may remain underrepresented.  
- May still produce false positives/negatives in ambiguous cases.

---

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/wkwk08/Nudity-Detection-Android-Train-Model.git
cd Nudity-Detection-Android-Train-Model

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Dataset Preparation
```bash
python scripts/balance_split_datasets.py
```

### 2. Validation
```bash
python scripts/validate_splits.py
```

### 3. Training
```bash
python scripts/train_all_objectives.py
```
> **Note:** `.pth` files are the trained PyTorch models used for ML evaluation.

### 4. Evaluation
```bash
python scripts/evaluate_objectives.py
```

### 5. ONNX Re-Export (optional)
```bash
python scripts/reexport_to_onnx.py
```

### 6. Conversion to TFLite
```bash
python scripts/convert_all_to_tflite.py
```

### 7. Dashboard
```bash
streamlit run scripts/ui_all_objectives.py
```

---

## 🔗 Android Integration (Kotlin + TFLite)
After conversion, the `.tflite` models are deployed in our Android application written in Kotlin.  
This allows real-time nudity detection directly on mobile devices, without requiring a server backend.

**⚠️ Status:** The Android application is still ongoing and under active development.

- **Pipeline:**  
  `.pth` (PyTorch checkpoint) → `.onnx` (interoperable format) → `.tflite` (optimized for mobile)  
  
- **Usage in Kotlin:**  
  The `.tflite` models are loaded using TensorFlow Lite's Interpreter API in Kotlin.  
  The app processes camera or gallery images, runs inference locally, and returns detection results.  
  This ensures privacy (no cloud upload) and efficiency (runs on-device).  
  
- **Companion Repository:**  
  Full Android implementation is available here:  
  👉 [Nudity-Detection-Android-Enhanced](https://github.com/wkwk08/Nudity-Detection-Android-Enhanced)

> **Note:** The `.tflite` models are the final deployment artifacts. They are lightweight, mobile-optimized, and directly integrated into the Kotlin app for thesis demonstration.

---

## 📂 Repository Structure
```
Nudity-Detection-Android-Train-Model/
├── scripts/              
├── models/               
├── logs/                 
├── datasets/             
├── balanced_output/      
├── requirements.txt      
├── .env                  
├── .gitignore            
└── README.md             
```

---

## 📑 Citation

This work is part of an ongoing, unpublished undergraduate thesis defense.  
If referencing this project, please cite:

> Abbariao, J.E.A., Ambrocio, J.P., Amper, M.J.F. (2025).  
> *An Enhancement of Nudity Detection Algorithm for Android Mobile Applications.*  
> Pamantasan ng Lungsod ng Maynila, BSCS Thesis (Unpublished Undergraduate Thesis).

---

## 📧 Contact
For academic inquiries, please contact the repository owner, [wkwk08](https://github.com/wkwk08).

---

## 📄 License
This project is part of an academic thesis. Please contact the authors for usage permissions.

---

**© 2025 Pamantasan ng Lungsod ng Maynila**