# GitHub Push - Step by Step Commands for Vijay Bedage
# Animal Classification Project

# ─── STEP 1: Install Git (if not installed) ─────────────────────
sudo apt install git -y        # Linux
# or: brew install git         # Mac

# ─── STEP 2: Configure Git (one time only) ──────────────────────
git config --global user.name "Vijay Bedage"
git config --global user.email "your_email@gmail.com"

# ─── STEP 3: Go to your project folder ─────────────────────────
cd animal-classification        # the folder you downloaded

# ─── STEP 4: Initialize Git ──────────────────────────────────────
git init
git branch -M main

# ─── STEP 5: Add all files ───────────────────────────────────────
git add .

# ─── STEP 6: First commit ────────────────────────────────────────
git commit -m "Initial commit: Animal Classification using MobileNetV2 Transfer Learning

- 15 animal classes (Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, 
  Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra)
- ~1944 training images
- MobileNetV2 + custom classification head
- Streamlit web app for real-time inference
- EDA notebook with dataset analysis
- Training pipeline with early stopping and LR scheduling"

# ─── STEP 7: Create Repo on GitHub ──────────────────────────────
# Go to: https://github.com/new
# Repository name: animal-classification
# Description: Animal image classification using MobileNetV2 Transfer Learning | 15 classes | Streamlit App
# Set to: Public
# Do NOT check "Add README" (we already have one)
# Click: Create Repository

# ─── STEP 8: Connect and Push ───────────────────────────────────
git remote add origin https://github.com/YOUR_USERNAME/animal-classification.git
git push -u origin main

# ─── IF you get authentication error ────────────────────────────
# GitHub no longer accepts passwords. Use a Personal Access Token:
# 1. Go to: github.com → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
# 2. Click "Generate new token (classic)"
# 3. Give it a name, set expiry to 90 days
# 4. Check: repo, workflow
# 5. Click Generate → COPY the token
# 6. When git asks for password → paste the token

# ─── STEP 9: After Training (push model results) ────────────────
# After you train the model locally, push results:
git add results/
git commit -m "Add training results: accuracy plots and confusion matrix"
git push
