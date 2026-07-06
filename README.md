# 🐾 Animal Classification using Transfer Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A deep learning project to classify 15 animal species using MobileNetV2 Transfer Learning**

[Quick Start](#-quick-start-step-by-step) • [Web App](#-launch-web-app) • [Project Structure](#-project-structure) • [Results](#-results)

</div>

---

## 📌 Project Overview

This project uses **Transfer Learning** with **MobileNetV2** (pretrained on ImageNet) to classify images into **15 different animal categories**. It features:

- ✅ **Two-phase training pipeline** — Phase 1: frozen base → Phase 2: fine-tune top 30 layers
- ✅ **Centralized config module** — all settings in one place (`src/config.py`)
- ✅ **Streamlit web app** — upload images for real-time predictions
- ✅ **Prediction script** — classify images from command line
- ✅ **Unit tests** — validate configuration with `pytest`
- ✅ **Reproducible** — seed-controlled, pinned dependency versions

---

## 🐾 Animal Classes (15 Total)

| | | | | |
|---|---|---|---|---|
| 🐻 Bear | 🐦 Bird | 🐱 Cat | 🐄 Cow | 🦌 Deer |
| 🐶 Dog | 🐬 Dolphin | 🐘 Elephant | 🦒 Giraffe | 🐴 Horse |
| 🦘 Kangaroo | 🦁 Lion | 🐼 Panda | 🐯 Tiger | 🦓 Zebra |

---

## 🚀 Quick Start (Step by Step)

Follow these steps to clone, install, and run the project on your machine.

### Prerequisites

Make sure you have the following installed:

| Tool | Version | Check Command |
|---|---|---|
| **Python** | 3.10 or higher | `python --version` |
| **pip** | Latest | `pip --version` |
| **Git** | Any | `git --version` |

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Vijaybedage/ML-project.git
cd ML-project
```

---

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> 💡 You should see `(venv)` at the beginning of your terminal line after activation.

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
| Package | Version | Purpose |
|---|---|---|
| tensorflow | 2.15.0 | Deep learning framework |
| numpy | 1.24.3 | Numerical computing |
| scikit-learn | 1.3.2 | Metrics & evaluation |
| matplotlib | 3.8.2 | Plotting & visualization |
| seaborn | 0.13.0 | Statistical visualization |
| Pillow | 10.1.0 | Image loading & processing |
| streamlit | 1.29.0 | Web app framework |
| plotly | 5.18.0 | Interactive charts |
| jupyter | 1.0.0 | Notebook support |
| pytest | 7.4.3 | Unit testing |

---

### Step 4: Prepare the Dataset

Download or place your animal images in this folder structure:

```
ML-project/
└── Animal Classification/
    └── dataset/
        ├── Bear/        ← put bear images here
        ├── Bird/        ← put bird images here
        ├── Cat/
        ├── Cow/
        ├── Deer/
        ├── Dog/
        ├── Dolphin/
        ├── Elephant/
        ├── Giraffe/
        ├── Horse/
        ├── Kangaroo/
        ├── Lion/
        ├── Panda/
        ├── Tiger/
        └── Zebra/       ← 15 folders total
```

> ⚠️ Each folder should contain **JPG/JPEG/PNG** images of that animal. The training script automatically splits the data into 80% train / 20% validation.

---

### Step 5: Train the Model

```bash
python src/train.py
```

**What happens during training:**

| Phase | Description | Epochs | Learning Rate |
|---|---|---|---|
| **Phase 1** | Train classification head only (MobileNetV2 base frozen) | 30 | 0.001 |
| **Phase 2** | Fine-tune top 30 layers of MobileNetV2 | 10 | 0.00001 |

**Output files saved to `results/` folder:**
- `training_history.png` — accuracy & loss curves
- `confusion_matrix.png` — per-class prediction heatmap
- `classification_report.txt` — precision, recall, F1-score
- `history.json` — raw training metrics

**Trained model saved to:**
- `models/animal_classifier.h5`

---

### Step 6: Predict on a Single Image

```bash
python src/predict.py path/to/your/animal_image.jpg
```

**Example:**
```bash
python src/predict.py "Animal Classification/dataset/Tiger/tiger_001.jpg"
```

**With a custom model path:**
```bash
python src/predict.py path/to/image.jpg models/animal_classifier.h5
```

**Output:**
```
Predicted: 🐯 Tiger (97.3% confidence)
Top 3: Tiger (97.3%), Lion (1.8%), Cat (0.5%)
```

---

### Step 7: Launch Web App

```bash
streamlit run app.py
```

This opens a browser at `http://localhost:8501` where you can:
- 📤 Upload any animal image (JPG/PNG)
- 🔍 See real-time prediction with confidence score
- 📊 View top-5 predictions with interactive bar chart
- 🧠 Read fun animal facts for each prediction
- 🎨 Enjoy a clean, responsive UI

---

### Step 8: Run Unit Tests (Optional)

```bash
pytest tests/ -v
```

This validates that all 15 animal classes, emojis, fun facts, and config settings are correctly defined.

---

## 🏗️ Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (pretrained on ImageNet)
  → Phase 1: All layers frozen
  → Phase 2: Top 30 layers fine-tuned
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, ReLU) → Dropout(0.5)
    ↓
Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(15, Softmax)  ← Output
```

### Why MobileNetV2?
- ✅ Lightweight and fast — ideal for small datasets
- ✅ Pretrained on 1.2M ImageNet images
- ✅ Excellent accuracy with fine-tuning
- ✅ Mobile and edge deployment ready

---

## 📁 Project Structure

```
ML-project/
├── Animal Classification/
│   └── dataset/               # Dataset (15 animal folders)
├── models/
│   └── animal_classifier.h5   # Saved trained model
├── notebooks/
│   └── EDA_and_Analysis.ipynb # Exploratory data analysis
├── results/
│   ├── training_history.png   # Accuracy & loss plots
│   ├── confusion_matrix.png   # Confusion matrix heatmap
│   ├── classification_report.txt
│   └── history.json           # Raw training metrics
├── src/
│   ├── __init__.py            # Package init
│   ├── config.py              # ⭐ Centralized configuration
│   ├── train.py               # Training pipeline (2-phase)
│   └── predict.py             # Inference script
├── tests/
│   ├── __init__.py            # Package init
│   └── test_config.py         # Unit tests for config
├── app.py                     # Streamlit web app
├── requirements.txt           # Pinned dependencies
└── README.md                  # This file
```

---

## 📈 Training Configuration

All settings are centralized in [`src/config.py`](src/config.py):

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Phase 1 Epochs | 30 (with early stopping) |
| Phase 1 Learning Rate | 0.001 (Adam) |
| Phase 2 Epochs | 10 (fine-tuning) |
| Phase 2 Learning Rate | 0.00001 (Adam) |
| Fine-tune Layers | Top 30 of MobileNetV2 |
| Seed | 42 (reproducible) |
| Loss | Categorical Crossentropy |
| Data Augmentation | Rotation, Flip, Zoom, Shift |

---

## 🎯 Results

| Metric | Value |
|---|---|
| Validation Accuracy | ~93%+ |
| Model Size | ~14 MB |
| Inference Time | < 100ms per image |

---

## 📊 Dataset Info

| Property | Value |
|---|---|
| Total Images | ~1,944 |
| Classes | 15 |
| Images per class | ~122–137 |
| Train / Val Split | 80% / 20% |
| Image Format | JPG / JPEG / PNG |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Programming Language |
| TensorFlow / Keras | Deep Learning Framework |
| MobileNetV2 | Pretrained Base Model |
| scikit-learn | Metrics & Evaluation |
| Streamlit | Web Application |
| Plotly | Interactive Charts |
| Matplotlib / Seaborn | Visualizations |
| Pillow | Image Processing |
| pytest | Unit Testing |

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `Model not found` error | Train the model first: `python src/train.py` |
| `No module named 'config'` | Run scripts from the project root folder (`ML-project/`) |
| CUDA/GPU errors | TensorFlow works on CPU too — just slower. Ignore GPU warnings |
| Streamlit not opening | Try `http://localhost:8501` manually in browser |
| Permission denied (Git push) | Use a GitHub Personal Access Token instead of password |

---

## 🔄 Recent Changes (July 2026)

- ✅ **Added `src/config.py`** — centralized all classes, paths, hyperparameters, emojis, and fun facts
- ✅ **2-Phase Training** — Phase 1 trains head only, Phase 2 fine-tunes top 30 MobileNetV2 layers
- ✅ **Proper Logging** — replaced all `print()` with Python `logging` module
- ✅ **Reproducibility** — added seed control (`SEED=42`) for consistent results
- ✅ **Pinned Dependencies** — exact versions in `requirements.txt` to avoid compatibility issues
- ✅ **Unit Tests** — added `tests/test_config.py` with `pytest`
- ✅ **Better Docstrings** — all functions documented with Args/Returns/Raises
- ✅ **Input Validation** — predict.py checks if image/model files exist before running

---

## 👨‍💻 Author

**Vijay Bedage**  
MCA Student — KLE Technological University, Hubli  
- 🔗 [GitHub](https://github.com/Vijaybedage)  
- 📧 vijaybedage24@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
⭐ If you found this project helpful, please give it a star!
</div>
