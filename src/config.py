"""
Animal Classification - Shared Configuration
Author: Vijay Bedage

Central configuration file to eliminate duplication across modules.
All shared constants (classes, emojis, paths, hyperparameters) live here.
"""

import os

# ─── PATHS ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'Animal Classification', 'dataset')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'animal_classifier.h5')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')

# ─── MODEL CONFIG ─────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-5
FINE_TUNE_LAYERS = 30  # Number of top base-model layers to unfreeze
SEED = 42

# ─── CLASSES ──────────────────────────────────────────────────────────────────
CLASSES = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
    'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
    'Panda', 'Tiger', 'Zebra'
]

NUM_CLASSES = len(CLASSES)

# ─── EMOJIS ───────────────────────────────────────────────────────────────────
ANIMAL_EMOJIS = {
    'Bear': '🐻', 'Bird': '🐦', 'Cat': '🐱', 'Cow': '🐄',
    'Deer': '🦌', 'Dog': '🐶', 'Dolphin': '🐬', 'Elephant': '🐘',
    'Giraffe': '🦒', 'Horse': '🐴', 'Kangaroo': '🦘', 'Lion': '🦁',
    'Panda': '🐼', 'Tiger': '🐯', 'Zebra': '🦓'
}

# ─── FUN FACTS ────────────────────────────────────────────────────────────────
ANIMAL_FACTS = {
    'Bear': 'Bears can smell food from 20 miles away!',
    'Bird': 'There are over 10,000 known species of birds.',
    'Cat': 'Cats sleep 12–16 hours a day.',
    'Cow': 'Cows have almost 360° panoramic vision.',
    'Deer': 'Deer can run up to 30 mph.',
    'Dog': 'Dogs have 300 million olfactory receptors in their nose.',
    'Dolphin': 'Dolphins sleep with one eye open.',
    'Elephant': 'Elephants are the only animals that can\'t jump.',
    'Giraffe': 'A giraffe\'s tongue is 18–20 inches long.',
    'Horse': 'Horses can sleep both standing up and lying down.',
    'Kangaroo': 'Kangaroos can\'t walk backwards.',
    'Lion': 'A lion\'s roar can be heard from 5 miles away.',
    'Panda': 'Giant pandas eat up to 38 pounds of bamboo a day.',
    'Tiger': 'No two tigers have the same stripe pattern.',
    'Zebra': 'Zebra stripes act as a natural bug repellent.'
}
