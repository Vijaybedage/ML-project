"""
Prediction Validation with Advanced OOD Detection & Temperature Scaling
Author: Vijay Bedage

Validates model predictions using:
- Confidence thresholding
- Margin analysis (top-1 vs top-2)
- Entropy-based uncertainty
- Mahalanobis distance OOD detection
- Temperature scaling for calibration
- Test-time augmentation (TTA) for robust predictions
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

# ─── VALIDATION THRESHOLDS ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60      # Minimum confidence (after temperature scaling)
MARGIN_THRESHOLD = 0.20          # Minimum gap between top-1 and top-2
ENTROPY_THRESHOLD = 2.0          # Maximum entropy (on original probs, higher = more confident)
OOD_DISTANCE_THRESHOLD = 40.0    # Maximum Mahalanobis distance
TTA_SAMPLES = 5                  # Number of TTA augmentations
TTA_CONSISTENCY_THRESHOLD = 0.15 # Maximum std across TTA samples

# Try to import OOD detector
try:
    from ood_detector import compute_ood_distance
    OOD_AVAILABLE = True
except ImportError:
    OOD_AVAILABLE = False
    def compute_ood_distance(features):
        return None


def compute_entropy(probs):
    """Compute Shannon entropy in bits."""
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log2(probs))


def temperature_scale(logits, temperature=1.5):
    """
    Apply temperature scaling to calibrate confidence.
    Temperature > 1 softens probabilities, reducing overconfidence.
    """
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)


def apply_tta(model, image_arr, num_samples=5):
    """
    Test-Time Augmentation: Run prediction on multiple augmented versions
    and average the probabilities for more robust predictions.
    
    Args:
        model: Keras model
        image_arr: Preprocessed image array (1, 224, 224, 3)
        num_samples: Number of augmented predictions to average
        
    Returns:
        Averaged probabilities, std across TTA samples
    """
    from tensorflow.keras.preprocessing.image import (
        random_rotation, random_shift, random_zoom, random_brightness
    )
    
    base_probs = model.predict(image_arr, verbose=0)[0]
    all_probs = [base_probs]
    
    for _ in range(num_samples - 1):
        aug_img = image_arr[0].copy()
        
        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        aug_img = random_rotation(aug_img, angle, row_axis=0, col_axis=1, channel_axis=2)
        
        # Random shift
        shift_x = np.random.uniform(-0.1, 0.1) * 224
        shift_y = np.random.uniform(-0.1, 0.1) * 224
        aug_img = random_shift(aug_img, shift_x/224, shift_y/224, row_axis=0, col_axis=1, channel_axis=2)
        
        # Random zoom
        zoom = np.random.uniform(0.9, 1.1)
        aug_img = random_zoom(aug_img, (zoom, zoom), row_axis=0, col_axis=1, channel_axis=2)
        
        # Random brightness
        brightness = np.random.uniform(0.9, 1.1)
        aug_img = random_brightness(aug_img, brightness)
        
        # Ensure valid range
        aug_img = np.clip(aug_img, 0, 1)
        aug_img = np.expand_dims(aug_img, axis=0)
        
        probs = model.predict(aug_img, verbose=0)[0]
        all_probs.append(probs)
    
    all_probs = np.array(all_probs)
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)
    
    return mean_probs, std_probs


def validate_prediction(probs, image_arr=None, image_features=None, model=None):
    """
    Validate a prediction using multiple criteria with TTA.
    
    Args:
        probs: Probability array (num_classes,)
        image_arr: Optional preprocessed image for TTA
        image_features: Optional precomputed features for OOD detection
        model: Optional model for TTA
        
    Returns:
        Dictionary with validation results
    """
    probs = np.asarray(probs).flatten()
    
    # Apply temperature scaling for calibration
    logits = np.log(probs + 1e-10)
    calibrated_probs = temperature_scale(logits, temperature=1.5)
    
    # Run TTA if model and image provided
    tta_std = None
    if model is not None and image_arr is not None:
        try:
            calibrated_probs, tta_std = apply_tta(model, image_arr)
        except Exception as e:
            logger.warning(f"TTA failed: {e}")
    
    # Top-2 predictions
    top2_idx = np.argsort(calibrated_probs)[::-1][:2]
    top1_idx, top2_idx = top2_idx[0], top2_idx[1]
    
    confidence = float(calibrated_probs[top1_idx])
    top2_prob = float(calibrated_probs[top2_idx])
    margin = confidence - top2_prob
    entropy = float(compute_entropy(probs))  # Use original probs for entropy
    
    # TTA uncertainty (if available)
    tta_uncertainty = float(tta_std[top1_idx]) if tta_std is not None else None
    
    # OOD distance
    ood_distance = None
    if OOD_AVAILABLE and image_features is not None:
        try:
            ood_distance = compute_ood_distance(image_features)
        except Exception as e:
            logger.warning(f"OOD detection failed: {e}")
    
    # ─── Validation Checks ─────────────────────────────────────────────────────
    checks = []
    
    if confidence < CONFIDENCE_THRESHOLD:
        checks.append(f"Low confidence: {confidence*100:.1f}% < {CONFIDENCE_THRESHOLD*100:.0f}%")
    
    if margin < MARGIN_THRESHOLD:
        top2_name = CLASSES[top2_idx] if top2_idx < len(CLASSES) else 'N/A'
        checks.append(f"Small margin: {margin*100:.1f}% < {MARGIN_THRESHOLD*100:.0f}% (top-2: {top2_name} at {top2_prob*100:.1f}%)")
    
    if entropy > ENTROPY_THRESHOLD:
        checks.append(f"High entropy: {entropy:.2f} bits > {ENTROPY_THRESHOLD:.1f} bits")
    
    if tta_uncertainty is not None and tta_uncertainty > TTA_CONSISTENCY_THRESHOLD:
        checks.append(f"Inconsistent TTA: std={tta_uncertainty:.3f} > {TTA_CONSISTENCY_THRESHOLD:.2f}")
    
    if ood_distance is not None and ood_distance > OOD_DISTANCE_THRESHOLD:
        checks.append(f"OOD detected: Mahalanobis distance {ood_distance:.1f} > {OOD_DISTANCE_THRESHOLD:.1f}")
    
    is_valid = len(checks) == 0
    reason = "; ".join(checks) if checks else None
    
    return {
        'is_valid': is_valid,
        'reason': reason,
        'confidence': confidence,
        'margin': margin,
        'entropy': entropy,
        'ood_distance': ood_distance,
        'tta_uncertainty': tta_uncertainty,
        'top1_class': CLASSES[top1_idx],
        'top2_class': CLASSES[top2_idx] if top2_idx < len(CLASSES) else 'N/A',
        'top1_idx': int(top1_idx),
        'top2_idx': int(top2_idx),
        'top2_prob': top2_prob,
        'calibrated_probs': calibrated_probs.tolist()
    }


def format_rejection_message(validation):
    """Format a user-friendly rejection message."""
    if validation['is_valid']:
        return None
    
    top1 = validation['top1_class']
    top2 = validation['top2_class']
    conf = validation['confidence'] * 100
    margin = validation['margin'] * 100
    top2_prob = validation.get('top2_prob', 0) * 100
    entropy = validation['entropy']
    ood_dist = validation.get('ood_distance')
    tta_unc = validation.get('tta_uncertainty')
    
    msg = "[!] Cannot Confidently Classify This Image\n\n"
    msg += f"Reason: {validation['reason']}\n\n"
    msg += "Details:\n"
    msg += f"- Top prediction: {top1} ({conf:.1f}%)\n"
    msg += f"- Second best: {top2} ({top2_prob:.1f}%)\n"
    msg += f"- Confidence gap: {margin:.1f}%\n"
    msg += f"- Prediction entropy: {entropy:.2f} bits\n"
    
    if tta_unc is not None:
        msg += f"- TTA consistency: {tta_unc:.3f}\n"
    
    if ood_dist is not None:
        msg += f"- OOD distance: {ood_dist:.1f}\n"
    
    msg += f"\nThis model recognizes 15 animals: {', '.join(CLASSES)}\n"
    msg += "\nPlease upload a clear photo of one of these animals."
    
    return msg


def get_validation_thresholds():
    """Return the current validation thresholds for display/debugging."""
    return {
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'margin_threshold': MARGIN_THRESHOLD,
        'entropy_threshold': ENTROPY_THRESHOLD,
        'ood_distance_threshold': OOD_DISTANCE_THRESHOLD,
        'tta_samples': TTA_SAMPLES,
        'tta_consistency_threshold': TTA_CONSISTENCY_THRESHOLD
    }


if __name__ == '__main__':
    # Test validation logic
    print("Testing validation logic...")
    
    # High confidence
    test_probs = np.array([0.9, 0.05, 0.03, 0.01, 0.01] + [0.0]*10)
    result = validate_prediction(test_probs)
    print(f"High confidence: {result['is_valid']} — {result.get('reason', 'OK')}")
    
    # Low confidence (close top-2)
    test_probs2 = np.array([0.4, 0.35, 0.15, 0.05, 0.05] + [0.0]*10)
    result2 = validate_prediction(test_probs2)
    print(f"Low confidence: {result2['is_valid']} — {result2.get('reason', 'OK')}")
    
    # High entropy (uniform)
    test_probs3 = np.ones(15) / 15
    result3 = validate_prediction(test_probs3)
    print(f"High entropy: {result3['is_valid']} — {result3.get('reason', 'OK')}")
    
    # Test rejection message
    if not result2['is_valid']:
        print("\n" + format_rejection_message(result2))