"""
Prediction Validation with OOD Detection
Author: Vijay Bedage

Validates model predictions using confidence thresholds, margin analysis,
entropy, and out-of-distribution detection.
"""

import os
import sys
import numpy as np
import logging

# Add parent directory for config import
sys.path.insert(0, os.path.dirname(__file__))
from config import CLASSES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─── VALIDATION THRESHOLDS ────────────────────────────────────────────────────
# These can be tuned based on validation set performance
CONFIDENCE_THRESHOLD = 0.50   # Minimum max probability
MARGIN_THRESHOLD = 0.15       # Minimum gap between top-1 and top-2
ENTROPY_THRESHOLD = 1.5       # Maximum entropy (bits) for confident prediction
OOD_DISTANCE_THRESHOLD = 50.0 # Maximum Mahalanobis distance (adjust after computing stats)

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


def validate_prediction(probs, image_arr=None, image_features=None):
    """
    Validate a prediction using multiple criteria.
    
    Args:
        probs: Probability array of shape (num_classes,).
        image_arr: Optional preprocessed image array for OOD detection.
        image_features: Optional precomputed features for OOD detection.
        
    Returns:
        Dictionary with validation results:
        - is_valid: Boolean indicating if prediction is reliable
        - reason: String explaining rejection (if invalid)
        - confidence: Top-1 probability
        - margin: Gap between top-1 and top-2
        - entropy: Shannon entropy in bits
        - ood_distance: Mahalanobis distance (if available)
    """
    probs = np.asarray(probs).flatten()
    
    # Top-2 predictions
    top2_idx = np.argsort(probs)[::-1][:2]
    top1_idx, top2_idx = top2_idx[0], top2_idx[1]
    
    confidence = float(probs[top1_idx])
    top2_prob = float(probs[top2_idx])
    margin = confidence - top2_prob
    entropy = float(compute_entropy(probs))
    
    # OOD distance
    ood_distance = None
    if OOD_AVAILABLE:
        if image_features is not None:
            ood_distance = compute_ood_distance(image_features)
        elif image_arr is not None:
            try:
                from ood_detector import extract_features
                features = extract_features(image_arr)
                ood_distance = compute_ood_distance(features)
            except Exception as e:
                logger.warning(f"OOD detection failed: {e}")
    
    # ─── Validation Checks ─────────────────────────────────────────────────────
    checks = []
    
    if confidence < CONFIDENCE_THRESHOLD:
        checks.append(f"Low confidence: {confidence*100:.1f}% < {CONFIDENCE_THRESHOLD*100:.0f}%")
    
    if margin < MARGIN_THRESHOLD:
        checks.append(f"Small margin: {margin*100:.1f}% < {MARGIN_THRESHOLD*100:.0f}% (top-2: {CLASSES[top2_idx]} at {top2_prob*100:.1f}%)")
    
    if entropy > ENTROPY_THRESHOLD:
        checks.append(f"High entropy: {entropy:.2f} bits > {ENTROPY_THRESHOLD:.1f} bits")
    
    if ood_distance is not None and ood_distance > OOD_DISTANCE_THRESHOLD:
        checks.append(f"OOD distance: {ood_distance:.1f} > {OOD_DISTANCE_THRESHOLD:.1f}")
    
    # ─── Result ────────────────────────────────────────────────────────────────
    if checks:
        reason = "; ".join(checks)
        return {
            'is_valid': False,
            'reason': reason,
            'confidence': confidence,
            'margin': margin,
            'entropy': entropy,
            'ood_distance': ood_distance,
            'top1_class': CLASSES[top1_idx],
            'top2_class': CLASSES[top2_idx],
            'top2_prob': top2_prob
        }
    
    return {
        'is_valid': True,
        'reason': None,
        'confidence': confidence,
        'margin': margin,
        'entropy': entropy,
        'ood_distance': ood_distance,
        'top1_class': CLASSES[top1_idx],
        'top2_class': CLASSES[top2_idx],
        'top2_prob': top2_prob
    }


def get_validation_thresholds():
    """Return the current validation thresholds for display/debugging."""
    return {
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'margin_threshold': MARGIN_THRESHOLD,
        'entropy_threshold': ENTROPY_THRESHOLD,
        'ood_distance_threshold': OOD_DISTANCE_THRESHOLD
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
    
    msg = "[!] Not a Recognized Animal\n\n"
    msg += f"Reason: {validation['reason']}\n\n"
    msg += "Details:\n"
    msg += f"- Top prediction: {top1} ({conf:.1f}%)\n"
    msg += f"- Second best: {top2} ({top2_prob:.1f}%)\n"
    msg += f"- Confidence gap: {margin:.1f}%\n"
    msg += f"- Prediction entropy: {entropy:.2f} bits\n"
    
    if ood_dist is not None:
        msg += f"- OOD distance: {ood_dist:.1f}\n"
    
    msg += f"\nThis model recognizes 15 animals: {', '.join(CLASSES)}\n"
    msg += "\nPlease upload an image of one of these animals."
    
    return msg


if __name__ == '__main__':
    # Test validation
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