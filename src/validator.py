"""
Prediction Validation with Advanced OOD Detection & Temperature Scaling
Author: Vijay Bedage

Validates model predictions using a 3-tier approach:
  Tier 1 (Auto-Accept):  High confidence + wide margin → accept immediately
  Tier 2 (Auto-Reject):  Very low confidence / high entropy → "Not an Animal"
  Tier 3 (Full Check):   Moderate zone → multi-criteria validation

Rejection types:
  - 'not_animal': Image is clearly not one of the 15 supported animals
  - 'uncertain':  Looks like an animal but model is unsure which one
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

# ─── TIER 1: AUTO-ACCEPT THRESHOLDS ──────────────────────────────────────────
# If the model is THIS confident, accept regardless of OOD distance
HIGH_CONFIDENCE = 0.85           # ≥85% confidence → auto-accept
HIGH_CONFIDENCE_MARGIN = 0.15    # AND ≥15% gap to second-best

# ─── TIER 2: AUTO-REJECT THRESHOLDS ─────────────────────────────────────────
# If the model is THIS uncertain, reject as "not an animal"
LOW_CONFIDENCE = 0.40            # <40% → probably not a known animal
VERY_HIGH_ENTROPY = 3.0          # >3.0 bits → extremely uncertain

# ─── TIER 3: MODERATE ZONE THRESHOLDS ────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60      # Minimum confidence for moderate zone
MARGIN_THRESHOLD = 0.15          # Minimum gap between top-1 and top-2
ENTROPY_THRESHOLD = 2.0          # Maximum entropy
OOD_DISTANCE_THRESHOLD = 50.0    # Maximum Mahalanobis distance (relaxed from 40)
TTA_SAMPLES = 5                  # Number of TTA augmentations
TTA_CONSISTENCY_THRESHOLD = 0.15 # Maximum std across TTA samples

# Minimum checks that must fail for moderate-zone rejection
MIN_FAILURES_FOR_REJECTION = 2   # At least 2 checks must fail

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
    Validate a prediction using 3-tier smart validation.
    
    Tier 1: Auto-accept if confidence is very high (e.g., Lion at 100%)
    Tier 2: Auto-reject as "not_animal" if confidence is very low
    Tier 3: Full multi-criteria check for moderate cases
    
    Args:
        probs: Probability array (num_classes,)
        image_arr: Optional preprocessed image for TTA
        image_features: Optional precomputed features for OOD detection
        model: Optional model for TTA
        
    Returns:
        Dictionary with validation results including 'rejection_type'
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
    top1_idx, top2_idx_val = top2_idx[0], top2_idx[1]
    
    confidence = float(calibrated_probs[top1_idx])
    top2_prob = float(calibrated_probs[top2_idx_val])
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
    
    # Build the base result dict
    result = {
        'confidence': confidence,
        'margin': margin,
        'entropy': entropy,
        'ood_distance': ood_distance,
        'tta_uncertainty': tta_uncertainty,
        'top1_class': CLASSES[top1_idx],
        'top2_class': CLASSES[top2_idx_val] if top2_idx_val < len(CLASSES) else 'N/A',
        'top1_idx': int(top1_idx),
        'top2_idx': int(top2_idx_val),
        'top2_prob': top2_prob,
        'calibrated_probs': calibrated_probs.tolist()
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 1: AUTO-ACCEPT — High confidence predictions
    # ═══════════════════════════════════════════════════════════════════════════
    if confidence >= HIGH_CONFIDENCE and margin >= HIGH_CONFIDENCE_MARGIN:
        logger.info(
            f"Tier 1 AUTO-ACCEPT: {CLASSES[top1_idx]} at {confidence*100:.1f}% "
            f"(margin={margin*100:.1f}%)"
        )
        result.update({
            'is_valid': True,
            'rejection_type': None,
            'reason': None,
            'tier': 1
        })
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 2: AUTO-REJECT — Clearly not a known animal
    # ═══════════════════════════════════════════════════════════════════════════
    is_very_low_confidence = confidence < LOW_CONFIDENCE
    is_very_high_entropy = entropy > VERY_HIGH_ENTROPY
    
    # Also check: if BOTH low confidence AND high OOD distance → not an animal
    is_ood_with_low_conf = (
        ood_distance is not None
        and ood_distance > OOD_DISTANCE_THRESHOLD
        and confidence < CONFIDENCE_THRESHOLD
    )
    
    if is_very_low_confidence or is_very_high_entropy or is_ood_with_low_conf:
        reasons = []
        if is_very_low_confidence:
            reasons.append(f"Very low confidence: {confidence*100:.1f}% < {LOW_CONFIDENCE*100:.0f}%")
        if is_very_high_entropy:
            reasons.append(f"Very high entropy: {entropy:.2f} bits > {VERY_HIGH_ENTROPY:.1f} bits")
        if is_ood_with_low_conf:
            reasons.append(f"Out-of-distribution with low confidence")
        
        logger.info(f"Tier 2 AUTO-REJECT (not_animal): {'; '.join(reasons)}")
        result.update({
            'is_valid': False,
            'rejection_type': 'not_animal',
            'reason': '; '.join(reasons),
            'tier': 2
        })
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 3: MODERATE ZONE — Full multi-criteria validation
    # Multiple checks must fail before rejection
    # ═══════════════════════════════════════════════════════════════════════════
    checks = []
    
    if confidence < CONFIDENCE_THRESHOLD:
        checks.append(f"Low confidence: {confidence*100:.1f}% < {CONFIDENCE_THRESHOLD*100:.0f}%")
    
    if margin < MARGIN_THRESHOLD:
        top2_name = CLASSES[top2_idx_val] if top2_idx_val < len(CLASSES) else 'N/A'
        checks.append(
            f"Small margin: {margin*100:.1f}% < {MARGIN_THRESHOLD*100:.0f}% "
            f"(top-2: {top2_name} at {top2_prob*100:.1f}%)"
        )
    
    if entropy > ENTROPY_THRESHOLD:
        checks.append(f"High entropy: {entropy:.2f} bits > {ENTROPY_THRESHOLD:.1f} bits")
    
    if tta_uncertainty is not None and tta_uncertainty > TTA_CONSISTENCY_THRESHOLD:
        checks.append(f"Inconsistent TTA: std={tta_uncertainty:.3f} > {TTA_CONSISTENCY_THRESHOLD:.2f}")
    
    if ood_distance is not None and ood_distance > OOD_DISTANCE_THRESHOLD:
        checks.append(f"OOD distance: {ood_distance:.1f} > {OOD_DISTANCE_THRESHOLD:.1f}")
    
    # Require multiple failures for rejection in the moderate zone
    is_valid = len(checks) < MIN_FAILURES_FOR_REJECTION
    
    if is_valid:
        logger.info(
            f"Tier 3 ACCEPT: {CLASSES[top1_idx]} at {confidence*100:.1f}% "
            f"({len(checks)} check(s) failed, need {MIN_FAILURES_FOR_REJECTION} for rejection)"
        )
    else:
        logger.info(
            f"Tier 3 REJECT (uncertain): {CLASSES[top1_idx]} at {confidence*100:.1f}% "
            f"({len(checks)} checks failed: {'; '.join(checks)})"
        )
    
    result.update({
        'is_valid': is_valid,
        'rejection_type': 'uncertain' if not is_valid else None,
        'reason': '; '.join(checks) if checks else None,
        'tier': 3
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
    """Format message for images that are clearly not animals."""
    msg = "❌ Not a Recognized Animal\n\n"
    msg += "This image does not appear to be one of the 15 animals\n"
    msg += "that this model is trained to recognize.\n\n"
    msg += f"Supported animals:\n"
    msg += f"{', '.join(CLASSES)}\n\n"
    msg += "Please upload a clear photo of one of these animals."
    return msg


def _format_uncertain_message(validation):
    """Format message for uncertain animal predictions."""
    top1 = validation['top1_class']
    top2 = validation['top2_class']
    conf = validation['confidence'] * 100
    top2_prob = validation.get('top2_prob', 0) * 100
    margin = validation['margin'] * 100
    entropy = validation['entropy']
    ood_dist = validation.get('ood_distance')
    tta_unc = validation.get('tta_uncertainty')
    
    msg = "⚠️ Uncertain Classification\n\n"
    msg += "The model is not fully confident about this image.\n\n"
    msg += "Possible matches:\n"
    msg += f"  1. {top1} ({conf:.1f}%)\n"
    msg += f"  2. {top2} ({top2_prob:.1f}%)\n\n"
    msg += "Details:\n"
    msg += f"  • Confidence gap: {margin:.1f}%\n"
    msg += f"  • Prediction entropy: {entropy:.2f} bits\n"
    
    if tta_unc is not None:
        msg += f"  • TTA consistency: {tta_unc:.3f}\n"
    
    if ood_dist is not None:
        msg += f"  • OOD distance: {ood_dist:.1f}\n"
    
    msg += "\nTip: Try uploading a clearer, well-lit photo of the animal."
    return msg


def get_validation_thresholds():
    """Return the current validation thresholds for display/debugging."""
    return {
        'high_confidence': HIGH_CONFIDENCE,
        'high_confidence_margin': HIGH_CONFIDENCE_MARGIN,
        'low_confidence': LOW_CONFIDENCE,
        'very_high_entropy': VERY_HIGH_ENTROPY,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'margin_threshold': MARGIN_THRESHOLD,
        'entropy_threshold': ENTROPY_THRESHOLD,
        'ood_distance_threshold': OOD_DISTANCE_THRESHOLD,
        'tta_samples': TTA_SAMPLES,
        'tta_consistency_threshold': TTA_CONSISTENCY_THRESHOLD,
        'min_failures_for_rejection': MIN_FAILURES_FOR_REJECTION
    }


if __name__ == '__main__':
    # Test validation logic
    print("Testing 3-tier validation logic...\n")
    
    # Tier 1: High confidence (should auto-accept)
    test_probs = np.array([0.98, 0.01, 0.005, 0.005] + [0.0]*11)
    result = validate_prediction(test_probs)
    print(f"Tier 1 (Lion at 98%): valid={result['is_valid']}, tier={result['tier']}")
    
    # Tier 2: Very low confidence (should auto-reject as not_animal)
    test_probs2 = np.array([0.2, 0.15, 0.15, 0.1, 0.1] + [0.06]*5 + [0.0]*5)
    result2 = validate_prediction(test_probs2)
    print(f"Tier 2 (Uniform 20%): valid={result2['is_valid']}, "
          f"type={result2.get('rejection_type')}, tier={result2['tier']}")
    
    # Tier 3: Moderate confidence (uncertain)
    test_probs3 = np.array([0.5, 0.3, 0.1, 0.05, 0.05] + [0.0]*10)
    result3 = validate_prediction(test_probs3)
    print(f"Tier 3 (Cat at 50%): valid={result3['is_valid']}, "
          f"type={result3.get('rejection_type')}, tier={result3['tier']}")
    
    # Test rejection messages
    if not result2['is_valid']:
        print(f"\n--- Not Animal Message ---\n{format_rejection_message(result2)}")
    if not result3['is_valid']:
        print(f"\n--- Uncertain Message ---\n{format_rejection_message(result3)}")