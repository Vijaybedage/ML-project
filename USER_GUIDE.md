# 🐾 Animal Classification Project: Complete 0-to-100% User Guide

Welcome to the **Animal Classification Project**! This guide covers everything you need to know to set up, train, run, test, and troubleshoot this machine learning project from scratch.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Prerequisites & System Setup](#-prerequisites--system-setup)
3. [Project Directory Structure](#-project-directory-structure)
4. [Setting Up the Virtual Environment](#-setting-up-the-virtual-environment)
5. [Installing Dependencies](#-installing-dependencies)
6. [Preparing the Dataset](#-preparing-the-dataset)
7. [Training the AI Model](#-training-the-ai-model)
8. [Running the Streamlit Web Application](#-running-the-streamlit-web-application)
9. [Running Predictions via Command Line](#-running-predictions-via-command-line)
10. [Running Automated Tests](#-running-automated-tests)
11. [Running the EDA Notebook](#-running-the-eda-notebook)
12. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 📌 Project Overview
This project uses **Transfer Learning** with a pre-trained **MobileNetV2** model (trained on ImageNet) to classify images into **15 different animal categories**:

| | | | | |
| :---: | :---: | :---: | :---: | :---: |
| 🐻 Bear | 🐦 Bird | 🐱 Cat | 🐄 Cow | 🦌 Deer |
| 🐶 Dog | 🐬 Dolphin | 🐘 Elephant | 🦒 Giraffe | 🐴 Horse |
| 🦘 Kangaroo | 🦁 Lion | 🐼 Panda | 🐯 Tiger | 🦓 Zebra |

The model uses a **two-phase training pipeline** to achieve high accuracy (~93%+) on validation datasets while keeping the final model lightweight (~14 MB), making it ideal for real-time web deployment.

---

## ⚙️ Prerequisites & System Setup
Before starting, check that you have the required programs installed on your computer:

| Tool | Required Version | How to Check | Installation Link |
| :--- | :--- | :--- | :--- |
| **Python** | 3.10 or higher | `python --version` | [python.org/downloads](https://www.python.org/downloads/) |
| **Pip** | Latest version | `pip --version` | Pre-bundled with Python |
| **Git** | Any version | `git --version` | [git-scm.com](https://git-scm.com/) |

> ⚠️ **Windows Users**: During Python installation, make sure to check the box that says **"Add Python to PATH"**.

---

## 📁 Project Directory Structure
Here is an overview of the key files in the repository:

```text
ML-project/
├── .venv/                     # Python virtual environment (contains installed libraries)
├── Animal Classification/     
│   └── dataset/               # Folder where animal image subfolders are stored
├── models/
│   └── animal_classifier.h5   # The pre-trained model file (~14 MB)
├── notebooks/
│   └── EDA_and_Analysis.ipynb # Jupyter Notebook for data analysis
├── results/                   # Generated evaluation graphs & reports
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── training_history.png
├── src/
│   ├── config.py              # Central configurations (classes, paths, learning rates)
│   ├── predict.py             # CLI prediction script
│   └── train.py               # Model training script
├── tests/
│   └── test_config.py         # Unit tests checking configuration integrity
├── app.py                     # Streamlit Web App entry point
├── requirements.txt           # Pinned library dependencies
└── USER_GUIDE.md              # This file!
```

---

## 🏗️ Setting Up the Virtual Environment
A virtual environment isolates this project's packages so they do not conflict with other Python projects on your computer.

### 1. Open Your Terminal & Navigate to the Project Folder
Open your terminal (PowerShell or CMD on Windows, Terminal on Mac/Linux) and navigate to the project root directory:
```bash
cd "C:\Users\vijay bedage\OneDrive\Desktop\IMP\animal_classification_project"
```

### 2. Create the Environment (If not already created)
Run the following command to create a folder named `.venv` containing your isolated Python environment:
```bash
python -m venv .venv
```

### 3. Activate the Environment
Depending on your terminal type, run the corresponding activation command:

*   **Windows (PowerShell - Recommended)**:
    ```powershell
    .\.venv\Scripts\activate
    ```
    *💡 Troubleshooting PowerShell:* If you get an error saying **"running scripts is disabled on this system"**, run this command first to temporarily authorize scripts for this terminal session:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    ```
    Then run the activation command `.\.venv\Scripts\activate` again.

*   **Windows (Command Prompt / CMD)**:
    ```cmd
    .venv\Scripts\activate.bat
    ```
*   **Mac / Linux / Git Bash**:
    ```bash
    source .venv/bin/activate
    ```

> 💡 **Successful Activation Indicator**: Once activated, your terminal prompt will show `(.venv)` at the beginning, like this:
> `(.venv) PS C:\Users\vijay bedage\OneDrive\Desktop\IMP\animal_classification_project>`

---

## 📦 Installing Dependencies
With your virtual environment activated, run the following command to install all required libraries:

```bash
pip install -r requirements.txt
```

This will download and configure:
*   `tensorflow`: Deep Learning Framework
*   `streamlit`: Web App Interface
*   `plotly`: Interactive Graphs
*   `numpy` & `pillow`: Image Processing & Math Utilities
*   `scikit-learn` & `matplotlib`: Metrics & Visualization Plots
*   `pytest`: Test Runner

---

## 🗂️ Preparing the Dataset
The dataset consists of animal images organized into folders. The layout must be structured exactly like this:

```text
ML-project/
└── Animal Classification/
    └── dataset/
        ├── Bear/          ← Bear images (.jpg, .jpeg, .png)
        ├── Bird/          ← Bird images
        ├── Cat/           ← Cat images
        ...
        └── Zebra/         ← Zebra images (15 folders total)
```

*   **Rule**: Each subfolder name must match the class list exactly, starting with a capital letter (e.g., `Deer`, not `deer`).
*   **Format**: Use `.jpg`, `.jpeg`, or `.png` images.
*   **Quantity**: Aim for at least 80–100 images per category for good results.

---

## 🧠 Training the AI Model
If you want to train the model from scratch on your dataset:

1.  Make sure your virtual environment is active: `(.venv)` should show in the terminal prompt.
2.  Run the training command:
    ```bash
    python src/train.py
    ```

### What happens during training?
*   **Data Preparation**: Automatically splits your images into **80% training** and **20% validation** sets.
*   **Phase 1 (Epochs 1-30)**: Base MobileNetV2 is frozen. Only the classification head is trained with a learning rate of `0.001` to learn general animal features.
*   **Phase 2 (Epochs 1-10)**: The top 30 layers of MobileNetV2 are unfrozen. The network is fine-tuned with a very low learning rate (`0.00001`) to tweak filters.
*   **Callbacks**: Utilizes *Early Stopping* to stop training early if validation accuracy plateaus, saving time and preventing overfitting.

### Outputs Saved to Disk:
*   `models/animal_classifier.h5`: The final trained model file.
*   `results/training_history.png`: Graph of accuracy and loss curves.
*   `results/confusion_matrix.png`: Heatmap showing classifications and errors.
*   `results/classification_report.txt`: Statistical metrics per animal class.

---

## 🌐 Running the Streamlit Web Application
To start the interactive, user-friendly web interface:

1.  Execute the Streamlit launch command in your terminal:
    ```bash
    streamlit run app.py
    ```
2.  Your default web browser should open automatically to:
    **`http://localhost:8501`**
    *(If it doesn't open automatically, copy and paste that address into your browser's URL bar).*
3.  **Using the App**:
    *   Click **"Browse files"** or drag and drop any animal image (JPG/PNG).
    *   The app will display the image, run it through the model, and print the top prediction with its confidence score.
    *   It will output an interactive **Plotly horizontal bar chart** showing the top 5 possibilities.
    *   It will show a **Fun Fact** about the predicted animal.
4.  **Stopping the App**:
    *   Go to the terminal window and press **`Ctrl + C`**.

---

## 🖥️ Running Predictions via Command Line
If you want to classify an image quickly without loading the web app:

```bash
python src/predict.py "path/to/your/image.jpg"
```

**Example**:
```bash
python src/predict.py "Animal Classification/dataset/Tiger/Tiger_1.jpg"
```

*   This will print out the top 3 predictions directly in the terminal window and open a Matplotlib pop-up window overlaying the prediction on the image.

---

## 🧪 Running Automated Tests
The project contains unit tests to ensure that changes do not break configurations (e.g. class mappings, directories, shapes).

Run the tests using:
```bash
pytest tests/ -v
```

All 15 assertions should pass:
```text
tests/test_config.py::TestClasses::test_exactly_15_classes PASSED
tests/test_config.py::TestConfig::test_img_size_values PASSED
...
```

---

## 📊 Running the EDA Notebook
To explore class distributions, sample images, and dataset statistics:

1.  Launch Jupyter Notebook in your active terminal:
    ```bash
    jupyter notebook notebooks/EDA_and_Analysis.ipynb
    ```
2.  Run the cells from top to bottom in your browser.

---

## 🛠️ Troubleshooting & FAQs

### ❓ PowerShell Error: "running scripts is disabled on this system"
*   **Reason**: Windows restricts running script files by default for safety.
*   **Fix**: Run this command to allow script execution for the current terminal window:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    ```
    Then run the activation command again.

### ❓ Command Not Found: "python" or "pip"
*   **Reason**: Python was not added to your system environment variables (PATH) during installation.
*   **Fix**: Reinstall Python and ensure you check the **"Add Python to PATH"** checkbox. On Mac/Linux, you might need to use `python3` and `pip3` instead.

### ❓ Error: "Model not found at models/animal_classifier.h5"
*   **Reason**: You are running prediction or starting the web app, but the trained model file is missing.
*   **Fix**: Either download the pre-trained `animal_classifier.h5` file and place it in the `models/` directory, or train the model yourself by running `python src/train.py`.

### ❓ Error: "ModuleNotFoundError: No module named 'tensorflow'"
*   **Reason**: You are running a script using your global Python interpreter instead of the virtual environment.
*   **Fix**: Ensure `(.venv)` is visible at the start of your terminal command line. If it is not, activate the environment using `.\.venv\Scripts\activate` first.

### ❓ How do I deactivate/exit the virtual environment?
*   Simply type the following command and press Enter:
    ```bash
    deactivate
    ```
