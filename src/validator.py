"""
Prediction Validation for Animal Classification
Author: Vijay Bedage

Simplified validation using model's own "Not_Animal" class prediction.
The model is trained with 16 classes (15 animals + Not_Animal), so it
directly predicts whether an image is a known animal or not.

Validation checks:
- If model predicts "Not_Animal" → reject
- If confidence is very low → flag as uncertain
- Temperature scaling for calibrated confidence
"""

import os
import sys
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASSES, NUM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─── THRESHOLDS ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.50      # Minimum confidence for a valid prediction
MARGIN_THRESHOLD = 0.10          # Minimum gap between top-1 and top-2
NOT_ANIMAL_CLASS = 'Not_Animal'  # Class name for non-animal images

# Animal-only classes (excludes Not_Animal)
ANIMAL_CLASSES = [c for c in CLASSES if c != NOT_ANIMAL_CLASS]


def compute_entropy(probs):
    """Compute Shannon entropy in bits."""
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log2(probs))


def temperature_scale(logits, temperature=1.5):
    """Apply temperature scaling to calibrate confidence."""
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)


def validate_prediction(probs, image_arr=None, image_features=None, model=None):
    """
    Validate a prediction using the model's own Not_Animal class.
    
    Logic:
      1. If top prediction is "Not_Animal" → reject as not_animal
      2. If Not_Animal is in top-2 AND confidence gap is small → uncertain
      3. If confidence is very low → uncertain
      4. Otherwise → accept
    
    Args:
        probs: Probability array (num_classes,)
        image_arr: (unused, kept for API compatibility)
        image_features: (unused, kept for API compatibility)
        model: (unused, kept for API compatibility)
        
    Returns:
        Dictionary with validation results
    """
    probs = np.asarray(probs).flatten()
    
    # Apply temperature scaling for calibration
    logits = np.log(probs + 1e-10)
    calibrated_probs = temperature_scale(logits, temperature=1.5)
    
    # Top predictions
    sorted_idx = np.argsort(calibrated_probs)[::-1]
    top1_idx = sorted_idx[0]
    top2_idx = sorted_idx[1]
    
    confidence = float(calibrated_probs[top1_idx])
    top2_prob = float(calibrated_probs[top2_idx])
    margin = confidence - top2_prob
    entropy = float(compute_entropy(probs))
    
    top1_class = CLASSES[top1_idx]
    top2_class = CLASSES[top2_idx] if top2_idx < len(CLASSES) else 'N/A'
    
    # Build base result
    result = {
        'confidence': confidence,
        'margin': margin,
        'entropy': entropy,
        'ood_distance': None,  # No longer used
        'tta_uncertainty': None,
        'top1_class': top1_class,
        'top2_class': top2_class,
        'top1_idx': int(top1_idx),
        'top2_idx': int(top2_idx),
        'top2_prob': top2_prob,
        'calibrated_probs': calibrated_probs.tolist()
    }
    
    # ─── CHECK 1: Model predicted "Not_Animal" ────────────────────────────
    if top1_class == NOT_ANIMAL_CLASS:
        logger.info(
            f"REJECT (not_animal): Model predicted Not_Animal at "
            f"{confidence*100:.1f}%"
        )
        result.update({
            'is_valid': False,
            'rejection_type': 'not_animal',
            'reason': f"Model classified as Not_Animal ({confidence*100:.1f}%)",
            'tier': 2
        })
        return result
    
    # ─── CHECK 2: Not_Animal is strong runner-up ──────────────────────────
    not_animal_idx = CLASSES.index(NOT_ANIMAL_CLASS) if NOT_ANIMAL_CLASS in CLASSES else -1
    if not_animal_idx >= 0:
        not_animal_prob = float(calibrated_probs[not_animal_idx])
        # If Not_Animal probability is close to top prediction
        if not_animal_prob > 0.25 and margin < 0.20:
            logger.info(
                f"REJECT (uncertain): {top1_class} at {confidence*100:.1f}% but "
                f"Not_Animal at {not_animal_prob*100:.1f}%"
            )
            result.update({
                'is_valid': False,
                'rejection_type': 'uncertain',
                'reason': (f"Uncertain: {top1_class} ({confidence*100:.1f}%) vs "
                          f"Not_Animal ({not_animal_prob*100:.1f}%)"),
                'tier': 3
            })
            return result
    
    # ─── CHECK 3: Very low confidence on any class ────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            f"REJECT (uncertain): {top1_class} at {confidence*100:.1f}% "
            f"< {CONFIDENCE_THRESHOLD*100:.0f}%"
        )
        result.update({
            'is_valid': False,
            'rejection_type': 'uncertain',
            'reason': (f"Low confidence: {top1_class} at {confidence*100:.1f}% "
                      f"(need ≥{CONFIDENCE_THRESHOLD*100:.0f}%)"),
            'tier': 3
        })
        return result
    
    # ─── ACCEPT ───────────────────────────────────────────────────────────
    logger.info(
        f"ACCEPT: {top1_class} at {confidence*100:.1f}% "
        f"(margin={margin*100:.1f}%)"
    )
    result.update({
        'is_valid': True,
        'rejection_type': None,
        'reason': None,
        'tier': 1
    })
    return result


def format_rejection_message(validation):
    """Format a user-friendly rejection message based on rejection type."""
    if validation['is_valid']:
        return None
    
    rejection_type = validation.get('rejection_type', 'uncertain')
    
    if rejection_type == 'not_animal':
        return _format_not_animal_message(validation)
    else:
        return _format_uncertain_message(validation)


def _format_not_animal_message(validation):
    """Format message for images detected as not animals."""
    try:
        msg = "[X] Not a Recognized Animal\n\n"
        msg += "This image does not appear to be one of the 15 animals\n"
        msg += "that this model is trained to recognize.\n\n"
        msg += f"Supported animals:\n"
        msg += f"{', '.join(ANIMAL_CLASSES)}\n\n"
        msg += "Please upload a clear photo of one of these animals."
    except UnicodeEncodeError:
        msg = "Not a Recognized Animal\n\n"
        msg += f"Supported: {', '.join(ANIMAL_CLASSES)}\n"
        msg += "Please upload a clear photo of one of these animals."
    return msg


def _format_uncertain_message(validation):
    """Format message for uncertain predictions -- no predictions shown."""
    try:
        msg = "[!] Uncertain Classification\n\n"
        msg += "The model is not fully confident about this image.\n"
        msg += "This may not be a clear photo of one of our supported animals.\n\n"
        msg += f"Supported animals:\n"
        msg += f"{', '.join(ANIMAL_CLASSES)}\n\n"
        msg += "Tip: Try uploading a clearer, well-lit photo of the animal."
    except UnicodeEncodeError:
        msg = "Uncertain Classification\n\n"
        msg += f"Supported: {', '.join(ANIMAL_CLASSES)}\n"
        msg += "Tip: Try uploading a clearer, well-lit photo."
    return msg


def get_validation_thresholds():
    """Return the current validation thresholds."""
    return {
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'margin_threshold': MARGIN_THRESHOLD,
        'not_animal_class': NOT_ANIMAL_CLASS
    }


if __name__ == '__main__':
    print("Testing simplified validation logic...\n")
    
    # Test 1: Clear animal prediction
    probs = np.zeros(16)
    probs[CLASSES.index('Lion')] = 0.95
    probs[CLASSES.index('Tiger')] = 0.03
    result = validate_prediction(probs)
    print(f"Lion at 95%: valid={result['is_valid']}, class={result['top1_class']}")
    
    # Test 2: Not_Animal prediction
    probs2 = np.zeros(16)
    probs2[CLASSES.index('Not_Animal')] = 0.80
    probs2[CLASSES.index('Lion')] = 0.10
    result2 = validate_prediction(probs2)
    print(f"Not_Animal at 80%: valid={result2['is_valid']}, type={result2['rejection_type']}")
    
    # Test 3: Low confidence
    probs3 = np.ones(16) / 16
    result3 = validate_prediction(probs3)
    print(f"Uniform: valid={result3['is_valid']}, type={result3['rejection_type']}")
    
    print("\n--- Messages ---")
    if not result2['is_valid']:
        print(format_rejection_message(result2))