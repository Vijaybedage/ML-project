"""
Animal Classification - Predict on new images
Author: Vijay Bedage

Usage:
    python src/predict.py <image_path> [model_path]
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

# Add parent directory for config import when running as script
sys.path.insert(0, os.path.dirname(__file__))
from config import CLASSES, ANIMAL_EMOJIS, IMG_SIZE, MODEL_PATH

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """Load a saved Keras model from disk.

    Args:
        model_path: Path to the .h5 or SavedModel file.

    Returns:
        Loaded Keras model.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            f"Train the model first with: python src/train.py"
        )
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a single image for model inference.

    Args:
        image_path: Path to the image file.

    Returns:
        Preprocessed numpy array with shape (1, 224, 224, 3), values in [0, 1].

    Raises:
        FileNotFoundError: If the image file doesn't exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: '{image_path}'")

    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model: tf.keras.Model, image_path: str) -> dict:
    """Predict the animal class for a given image.

    Args:
        model: Loaded Keras model.
        image_path: Path to the image file.

    Returns:
        Dictionary with keys:
            - predicted_class (str): Top predicted class name.
            - confidence (float): Confidence score for the top prediction.
            - top3 (list[dict]): Top 3 predictions with class and confidence.
            - all_probabilities (dict): All class probabilities.
    """
    arr = preprocess_image(image_path)
    probs = model.predict(arr, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    return {
        'predicted_class': CLASSES[top3_idx[0]],
        'confidence': float(probs[top3_idx[0]]),
        'top3': [
            {'class': CLASSES[i], 'confidence': float(probs[i])}
            for i in top3_idx
        ],
        'all_probabilities': {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    }


def visualize_prediction(image_path: str, result: dict, save_path: str = None):
    """Display image with prediction overlay and top-3 bar chart.

    Args:
        image_path: Path to the original image.
        result: Prediction result dictionary from predict().
        save_path: Optional path to save the visualization. Shows plot if None.
    """
    img = Image.open(image_path).convert('RGB')
    label = result['predicted_class']
    conf = result['confidence']
    emoji = ANIMAL_EMOJIS.get(label, '')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(img)
    ax1.axis('off')
    color = '#2ecc71' if conf > 0.8 else '#f39c12' if conf > 0.5 else '#e74c3c'
    ax1.set_title(f'{emoji} {label}  ({conf*100:.1f}%)',
                  fontsize=16, fontweight='bold', color=color, pad=12)

    # Bar chart for top-3
    top3 = result['top3']
    names = [f"{ANIMAL_EMOJIS.get(t['class'],'')} {t['class']}" for t in top3]
    confs = [t['confidence'] * 100 for t in top3]
    colors = [color, '#3498db', '#95a5a6']
    bars = ax2.barh(names[::-1], confs[::-1], color=colors[::-1], edgecolor='white')
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Top 3 Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    for bar, val in zip(bars, confs[::-1]):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Prediction visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH

    model = load_model(model_path)
    result = predict(model, image_path)

    emoji = ANIMAL_EMOJIS.get(result['predicted_class'], '')
    try:
        print(f"\n{emoji} Predicted: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print("\nTop 3 predictions:")
        for i, t in enumerate(result['top3'], 1):
            print(f"   {i}. {ANIMAL_EMOJIS.get(t['class'],'')} {t['class']}: {t['confidence']*100:.2f}%")
    except UnicodeEncodeError:
        print(f"\nPredicted: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print("\nTop 3 predictions:")
        for i, t in enumerate(result['top3'], 1):
            print(f"   {i}. {t['class']}: {t['confidence']*100:.2f}%")

    visualize_prediction(image_path, result)
