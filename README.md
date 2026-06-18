# 🐾 Animal Classification using Transfer Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A deep learning project to classify 15 animal species using MobileNetV2 Transfer Learning**

</div>

---

## 📌 Project Overview

This project uses **Transfer Learning** with **MobileNetV2** (pretrained on ImageNet) to classify images into 15 different animal categories. It includes a full training pipeline, prediction script, and an interactive **Streamlit web app** for real-time inference.

---

## 🐾 Animal Classes (15 Total)

| | | | | |
|---|---|---|---|---|
| 🐻 Bear | 🐦 Bird | 🐱 Cat | 🐄 Cow | 🦌 Deer |
| 🐶 Dog | 🐬 Dolphin | 🐘 Elephant | 🦒 Giraffe | 🐴 Horse |
| 🦘 Kangaroo | 🦁 Lion | 🐼 Panda | 🐯 Tiger | 🦓 Zebra |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total Images | ~1,944 |
| Classes | 15 |
| Images per class | ~122–137 |
| Train / Val Split | 80% / 20% |
| Image Format | JPG / JPEG / PNG |

---

## 🏗️ Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (pretrained on ImageNet, frozen)
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

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/vijaybedage/animal-classification.git
cd animal-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
Place your dataset in the following structure:
```
Animal Classification/
└── dataset/
    ├── Bear/
    ├── Bird/
    ├── Cat/
    ... (15 classes)
    └── Zebra/
```

### 4. Train the Model
```bash
python src/train.py
```

### 5. Predict on a Single Image
```bash
python src/predict.py path/to/image.jpg
```

### 6. Launch Web App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
animal-classification/
├── Animal Classification/
│   └── dataset/          # Dataset (15 animal folders)
├── models/
│   └── animal_classifier.h5   # Saved trained model
├── notebooks/
│   └── EDA_and_Analysis.ipynb # Exploratory data analysis
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── dataset_distribution.png
├── src/
│   ├── train.py          # Training script
│   └── predict.py        # Inference script
├── app.py                # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 📈 Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 30 (with early stopping) |
| Optimizer | Adam (lr=0.001) |
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

## 🖥️ Web App Features

- 📤 Upload any animal image (JPG/PNG)
- 🔍 Real-time prediction with confidence score
- 📊 Top-5 predictions with interactive bar chart
- 🧠 Fun animal facts for each prediction
- 🎨 Clean, responsive UI

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

---

## 👨‍💻 Author

**Vijay Bedage**  
MCA Student — KLE Technological University, Hubli  
- 🔗 [GitHub](https://github.com/vijaybedage)  
- 📧 vijaybedage24@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
⭐ If you found this project helpful, please give it a star!
</div>
