# 🐾 Animal Classification using Transfer Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A deep learning project to classify 15 animal species using MobileNetV2 Transfer Learning**

[Quick Start](#-complete-setup-guide-0-to-100) • [Web App](#step-7-launch-the-streamlit-web-app-) • [Project Structure](#-project-structure) • [Results](#-results)

</div>

---

## 📌 Project Overview

This project uses **Transfer Learning** with **MobileNetV2** (pretrained on ImageNet) to classify images into **15 different animal categories**. It features:

- ✅ **Two-phase training pipeline** — Phase 1: frozen base → Phase 2: fine-tune top 30 layers
- ✅ **Centralized config module** — all settings in one place (`src/config.py`)
- ✅ **Streamlit web app** — upload images for real-time predictions with interactive charts
- ✅ **Prediction script** — classify images from the command line
- ✅ **EDA notebook** — explore the dataset with visualizations
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

## 🚀 Complete Setup Guide (0 to 100)

> Follow every step below in order. By the end, you will have the project fully running on your machine — from cloning to training to launching the web app.

---

### Step 1: Check Prerequisites ✅

Before you start, make sure the following tools are installed on your computer.

| Tool | Required Version | How to Check | How to Install |
|---|---|---|---|
| **Python** | 3.10 or higher | `python --version` | [python.org/downloads](https://www.python.org/downloads/) |
| **pip** | Latest | `pip --version` | Comes with Python. Upgrade: `python -m pip install --upgrade pip` |
| **Git** | Any | `git --version` | [git-scm.com/downloads](https://git-scm.com/downloads) |

> ⚠️ **Windows Users**: During Python installation, make sure to check **"Add Python to PATH"**. If you miss this, Python commands won't work in your terminal.

> 💡 **Tip**: If `python` doesn't work but `python3` does (common on Mac/Linux), use `python3` everywhere below instead of `python`.

---

### Step 2: Clone the Repository 📥

Open a terminal (PowerShell on Windows, Terminal on Mac/Linux) and run:

```bash
git clone https://github.com/Vijaybedage/ML-project.git
```

Then navigate into the project folder:

```bash
cd ML-project
```

> 📁 You should now be inside the `ML-project` folder. Verify by checking the contents — you should see `app.py`, `requirements.txt`, `src/`, etc.

**Verify you are in the right folder:**

```bash
# Windows (PowerShell)
dir

# Mac / Linux
ls
```

You should see files like `app.py`, `requirements.txt`, `src/`, `models/`, etc.

---

### Step 3: Create a Virtual Environment 🏗️

A virtual environment keeps this project's dependencies isolated from other Python projects on your system.

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> ✅ **How to know it worked:** You should see `(venv)` at the beginning of your terminal prompt, like this:
> ```
> (venv) C:\Users\you\ML-project>
> (venv) user@mac:~/ML-project$
> ```

> ⚠️ **PowerShell Execution Policy Error?** If you see a "running scripts is disabled" error on Windows, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then try `venv\Scripts\activate` again.

---

### Step 4: Install Dependencies 📦

With the virtual environment activated (you should see `(venv)` in your prompt), install all required packages:

```bash
pip install -r requirements.txt
```

This will install:

| Package | Version | Purpose |
|---|---|---|
| tensorflow | >= 2.15.0 | Deep learning framework |
| numpy | >= 1.24.3 | Numerical computing |
| scikit-learn | >= 1.3.2 | Metrics & evaluation |
| matplotlib | >= 3.8.2 | Plotting & visualization |
| seaborn | >= 0.13.0 | Statistical visualization |
| Pillow | >= 10.1.0 | Image loading & processing |
| streamlit | >= 1.29.0 | Web app framework |
| plotly | >= 5.18.0 | Interactive charts |
| jupyter | >= 1.0.0 | Notebook support |
| pytest | >= 7.4.3 | Unit testing |

> ⏳ **This may take 5–10 minutes** depending on your internet speed (TensorFlow alone is ~500 MB).

**Verify installation:**
```bash
python -c "import tensorflow; print('TensorFlow', tensorflow.__version__)"
python -c "import streamlit; print('Streamlit', streamlit.__version__)"
```

Expected output:
```
TensorFlow 2.15.0
Streamlit 1.29.0
```

---

### Step 5: Prepare the Dataset 🗂️

The dataset is **not included** in the Git repository (it's too large). You need to download and place it manually.

#### Option A: Download from Kaggle (Recommended)

1. Search for **"Animal Image Dataset - 15 Classes"** on [kaggle.com](https://www.kaggle.com)
2. Download and extract the dataset
3. Place the animal image folders inside `Animal Classification/dataset/`

#### Option B: Use Your Own Images

Create the following folder structure inside the project:

```
ML-project/
└── Animal Classification/
    └── dataset/
        ├── Bear/          ← put bear images here (JPG/PNG)
        ├── Bird/          ← put bird images here
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
        └── Zebra/         ← 15 folders total
```

> ⚠️ **Important Rules:**
> - Each folder name **must match exactly** (capital first letter): `Bear`, `Bird`, `Cat`, etc.
> - Images must be **JPG, JPEG, or PNG** format
> - Aim for **~100+ images per class** for good accuracy
> - The training script **automatically splits** the data into 80% train / 20% validation

**Verify dataset structure:**
```bash
# Windows (PowerShell)
dir "Animal Classification\dataset"

# Mac / Linux
ls "Animal Classification/dataset/"
```

You should see 15 folders (Bear, Bird, Cat, … Zebra).

---

### Step 6: Train the Model 🧠

> 💡 **Skip this step** if a pre-trained model file already exists at `models/animal_classifier.h5` and you just want to run predictions. Jump to [Step 7](#step-7-launch-the-streamlit-web-app-).

With the virtual environment activated and dataset in place, run:

```bash
python src/train.py
```

#### What happens during training:

| Phase | What It Does | Epochs | Learning Rate |
|---|---|---|---|
| **Phase 1** | Trains the classification head only (MobileNetV2 base layers are frozen) | Up to 30 (early stopping) | 0.001 |
| **Phase 2** | Unfreezes top 30 MobileNetV2 layers and fine-tunes the whole model | Up to 10 (early stopping) | 0.00001 |

> ⏳ **Training time:** ~15–30 minutes on CPU, ~5–10 minutes with GPU.

#### Output files generated:

| File | Location | Description |
|---|---|---|
| `animal_classifier.h5` | `models/` | The trained model (~14 MB) |
| `training_history.png` | `results/` | Accuracy & loss curves for both phases |
| `confusion_matrix.png` | `results/` | Per-class prediction heatmap |
| `classification_report.txt` | `results/` | Precision, recall, F1-score per class |
| `history.json` | `results/` | Raw training metrics (JSON) |

#### Expected console output (summary):

```
Starting Animal Classification Training...
  Classes: 15
  Train samples: ~1555
  Val samples:   ~389

═══ Phase 1: Training classification head (base frozen) ═══
Epoch 1/30 — accuracy: 0.45 — val_accuracy: 0.68
...
Phase 1 — Val Accuracy: 0.8900 (89.00%)

═══ Phase 2: Fine-tuning top 30 base layers ═══
Epoch 1/10 — accuracy: 0.91 — val_accuracy: 0.93
...
Final Validation Accuracy: 0.9300 (93.00%)
Improvement from fine-tuning: +4.00%
Model saved to: models/animal_classifier.h5
```

---

### Step 7: Launch the Streamlit Web App 🌐

This is the main interactive interface — upload an animal image and get a prediction.

```bash
streamlit run app.py
```

#### What happens:
1. Your **default browser** opens automatically at `http://localhost:8501`
2. If it doesn't open, manually visit: **http://localhost:8501**

#### Web app features:
- 📤 **Upload** any animal image (JPG/JPEG/PNG)
- 🔍 **Real-time prediction** with confidence percentage
- 📊 **Top-5 predictions** displayed in an interactive Plotly bar chart
- 🧠 **Fun animal facts** shown for each prediction
- 🎨 Clean, responsive UI with gradient styling

#### To stop the web app:
Press `Ctrl + C` in the terminal where Streamlit is running.

---

### Step 8: Predict from Command Line (Alternative) 🖥️

If you prefer the command line over the web app:

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

**Example output:**
```
🐯 Predicted: Tiger
   Confidence: 97.30%

📊 Top 3 predictions:
   1. 🐯 Tiger: 97.30%
   2. 🦁 Lion: 1.80%
   3. 🐱 Cat: 0.50%
```

A matplotlib window will also pop up showing the image with a top-3 bar chart visualization.

---

### Step 9: Explore the Data (Optional) 📊

An EDA (Exploratory Data Analysis) notebook is included to help you visualize the dataset distribution, sample images, and class balance.

```bash
jupyter notebook notebooks/EDA_and_Analysis.ipynb
```

This opens Jupyter in your browser. Run the cells top to bottom to explore.

---

### Step 10: Run Unit Tests (Optional) 🧪

Validate that all configuration (class names, emojis, fun facts, paths) is correctly set up:

```bash
pytest tests/ -v
```

**Expected output:**
```
tests/test_config.py::test_classes_count PASSED
tests/test_config.py::test_emoji_mapping PASSED
tests/test_config.py::test_animal_facts PASSED
...
```

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
│   ├── dataset/               # Dataset (15 animal image folders)
│   └── Image Classification   # Project documentation PDF
│       of animals.pdf
├── models/
│   └── animal_classifier.h5   # Saved trained model (~14 MB)
├── notebooks/
│   └── EDA_and_Analysis.ipynb  # Exploratory data analysis notebook
├── results/
│   ├── training_history.png    # Accuracy & loss plots
│   ├── confusion_matrix.png    # Confusion matrix heatmap
│   ├── classification_report.txt  # Per-class metrics
│   ├── dataset_distribution.png   # Class distribution chart
│   └── history.json            # Raw training metrics
├── src/
│   ├── __init__.py             # Package init
│   ├── config.py               # ⭐ Centralized configuration
│   ├── train.py                # Training pipeline (2-phase)
│   └── predict.py              # Inference script
├── tests/
│   ├── __init__.py             # Package init
│   └── test_config.py          # Unit tests for config
├── app.py                      # 🌐 Streamlit web app
├── requirements.txt            # Pinned dependency versions
├── .gitignore                  # Files excluded from Git
└── README.md                   # This file
```

---

## 📈 Training Configuration

All settings are centralized in [`src/config.py`](src/config.py) — change values there to experiment:

| Parameter | Value | Description |
|---|---|---|
| Image Size | 224 × 224 | Input resolution for MobileNetV2 |
| Batch Size | 32 | Images per training batch |
| Phase 1 Epochs | 30 (with early stopping) | Train classification head only |
| Phase 1 Learning Rate | 0.001 (Adam) | Initial learning rate |
| Phase 2 Epochs | 10 (with early stopping) | Fine-tune top layers |
| Phase 2 Learning Rate | 0.00001 (Adam) | Lower LR for fine-tuning |
| Fine-tune Layers | Top 30 of MobileNetV2 | Layers unfrozen in Phase 2 |
| Seed | 42 | For reproducible results |
| Data Split | 80% train / 20% val | Automatic split |
| Data Augmentation | Rotation, Flip, Zoom, Shift | Applied only to training set |

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

| # | Problem | Solution |
|---|---|---|
| 1 | `python` command not found | Make sure Python is added to PATH. Try `python3` instead |
| 2 | `pip install` fails with permission error | Use `pip install --user -r requirements.txt` or ensure you're in the virtual environment |
| 3 | `ModuleNotFoundError: No module named 'xyz'` | Make sure your virtual environment is activated (`venv\Scripts\activate`) and run `pip install -r requirements.txt` again |
| 4 | `Model not found` error in web app | Train the model first: `python src/train.py`, or ensure `models/animal_classifier.h5` exists |
| 5 | `No module named 'config'` | Run scripts from the **project root folder** (`ML-project/`), not from inside `src/` |
| 6 | CUDA / GPU warnings in terminal | These are safe to ignore. TensorFlow works on CPU too — just slower |
| 7 | Streamlit not opening in browser | Manually open `http://localhost:8501` in your browser |
| 8 | `running scripts is disabled` (Windows PowerShell) | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| 9 | `Dataset not found` error during training | Verify the dataset is placed at `Animal Classification/dataset/` with 15 named subfolders |
| 10 | `Permission denied` on Git push | Use a GitHub Personal Access Token instead of a password |
| 11 | TensorFlow / numpy installation fails on Python 3.13 | The requirements use flexible versioning (`>=`). Upgrade pip (`python -m pip install --upgrade pip`) or use Python 3.10 / 3.11. |

---

## 📋 Quick Reference Commands

```bash
# ── Setup (one-time) ──────────────────────────────
git clone https://github.com/Vijaybedage/ML-project.git
cd ML-project
python -m venv venv
venv\Scripts\activate              # Windows PowerShell
# source venv/bin/activate          # Mac/Linux
pip install -r requirements.txt

# ── Run ───────────────────────────────────────────
python src/train.py                # Train the model
streamlit run app.py               # Launch web app
python src/predict.py image.jpg    # Predict from CLI

# ── Extras ────────────────────────────────────────
pytest tests/ -v                   # Run unit tests
jupyter notebook notebooks/EDA_and_Analysis.ipynb  # EDA
```

---

## 🔄 Recent Changes (July 2026)

- ✅ **Added `src/config.py`** — centralized all classes, paths, hyperparameters, emojis, and fun facts
- ✅ **2-Phase Training** — Phase 1 trains head only, Phase 2 fine-tunes top 30 MobileNetV2 layers
- ✅ **Proper Logging** — replaced all `print()` with Python `logging` module
- ✅ **Reproducibility** — added seed control (`SEED=42`) for consistent results
- ✅ **Flexible Dependencies** — relaxed `requirements.txt` to use `>=` constraints instead of strict `==` to support newer Python versions (such as Python 3.13) and prevent installation errors.
- ✅ **Unit Tests** — added `tests/test_config.py` with `pytest`
- ✅ **Better Docstrings** — all functions documented with Args/Returns/Raises
- ✅ **Input Validation** — predict.py checks if image/model files exist before running
- ✅ **Streamlit Fix** — resolved `use_container_width` compatibility issue with Streamlit 1.29

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
