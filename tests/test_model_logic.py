"""
Unit tests for prediction, validation, and OOD detection logic.
Tests model inference pipeline without requiring a trained model.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import CLASSES, NUM_CLASSES, IMG_SIZE
from validator import (
    validate_prediction, compute_entropy, temperature_scale,
    format_rejection_message, get_validation_thresholds, ANIMAL_CLASSES,
    NOT_ANIMAL_CLASS
)

# ─── VALIDATOR TESTS ─────────────────────────────────────────────────────────

class TestComputeEntropy:
    """Tests for entropy calculation."""

    def test_uniform_distribution_max_entropy(self):
        probs = np.ones(16) / 16
        entropy = compute_entropy(probs)
        assert entropy == pytest.approx(np.log2(16), abs=0.01)

    def test_deterministic_distribution_zero_entropy(self):
        probs = np.zeros(16)
        probs[0] = 1.0
        entropy = compute_entropy(probs)
        assert entropy == pytest.approx(0.0, abs=0.01)

    def test_entropy_positive(self):
        probs = np.array([0.5, 0.3, 0.2] + [0.0] * 13)
        entropy = compute_entropy(probs)
        assert entropy > 0

    def test_handles_zeros(self):
        probs = np.zeros(16)
        probs[0] = 1.0
        entropy = compute_entropy(probs)
        assert not np.isnan(entropy)


class TestTemperatureScaling:
    """Tests for temperature calibration."""

    def test_temperature_one_identity(self):
        probs = np.array([0.7, 0.2, 0.1] + [0.0] * 13)
        logits = np.log(probs + 1e-10)
        scaled = temperature_scale(logits, temperature=1.0)
        assert scaled[0] == pytest.approx(probs[0], abs=0.05)

    def test_high_temperature_flattens(self):
        probs = np.array([0.9, 0.05, 0.05] + [0.0] * 13)
        logits = np.log(probs + 1e-10)
        scaled = temperature_scale(logits, temperature=3.0)
        # High temp should make distribution more uniform
        assert scaled[0] < probs[0]

    def test_output_sums_to_one(self):
        probs = np.array([0.5, 0.3, 0.2] + [0.0] * 13)
        logits = np.log(probs + 1e-10)
        scaled = temperature_scale(logits, temperature=1.5)
        assert scaled.sum() == pytest.approx(1.0, abs=0.01)


class TestValidatePrediction:
    """Tests for prediction validation logic."""

    def test_clear_animal_prediction_accepted(self):
        probs = np.zeros(16)
        lion_idx = CLASSES.index('Lion')
        probs[lion_idx] = 0.95
        probs[CLASSES.index('Tiger')] = 0.03
        result = validate_prediction(probs)
        assert result['is_valid'] is True
        assert result['top1_class'] == 'Lion'

    def test_not_animal_prediction_rejected(self):
        probs = np.zeros(16)
        not_animal_idx = CLASSES.index('Not_Animal')
        probs[not_animal_idx] = 0.80
        probs[CLASSES.index('Lion')] = 0.10
        result = validate_prediction(probs)
        assert result['is_valid'] is False
        assert result['rejection_type'] == 'not_animal'

    def test_uniform_distribution_rejected(self):
        probs = np.ones(16) / 16
        result = validate_prediction(probs)
        assert result['is_valid'] is False
        assert result['rejection_type'] == 'uncertain'

    def test_result_contains_required_keys(self):
        probs = np.zeros(16)
        probs[CLASSES.index('Dog')] = 0.90
        result = validate_prediction(probs)
        required_keys = ['is_valid', 'rejection_type', 'confidence', 'margin',
                         'entropy', 'top1_class', 'top2_class', 'tier']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_confidence_in_valid_range(self):
        probs = np.random.dirichlet(np.ones(16))
        result = validate_prediction(probs)
        assert 0 <= result['confidence'] <= 1

    def test_margin_in_valid_range(self):
        probs = np.random.dirichlet(np.ones(16))
        result = validate_prediction(probs)
        assert 0 <= result['margin'] <= 1

    def test_entropy_non_negative(self):
        probs = np.random.dirichlet(np.ones(16))
        result = validate_prediction(probs)
        assert result['entropy'] >= 0

    def test_model_kwarg_accepted(self):
        """validate_prediction should accept model kwarg for API compatibility."""
        probs = np.zeros(16)
        probs[CLASSES.index('Cat')] = 0.90
        result = validate_prediction(probs, model=None)
        assert result['is_valid'] is True

    def test_not_animal_strong_runner_up_uncertain(self):
        """When Not_Animal is strong runner-up with small margin, should be uncertain."""
        probs = np.zeros(16)
        dog_idx = CLASSES.index('Dog')
        not_animal_idx = CLASSES.index('Not_Animal')
        probs[dog_idx] = 0.35
        probs[not_animal_idx] = 0.30
        probs[CLASSES.index('Cat')] = 0.25
        result = validate_prediction(probs)
        # Should be uncertain due to strong Not_Animal runner-up
        assert result['is_valid'] is False


class TestFormatRejectionMessage:
    """Tests for rejection message formatting."""

    def test_not_animal_message(self):
        validation = {
            'is_valid': False,
            'rejection_type': 'not_animal',
            'confidence': 0.8,
            'top1_class': 'Not_Animal'
        }
        msg = format_rejection_message(validation)
        assert msg is not None
        assert 'Not' in msg or 'not' in msg

    def test_uncertain_message(self):
        validation = {
            'is_valid': False,
            'rejection_type': 'uncertain',
            'confidence': 0.3,
            'top1_class': 'Dog'
        }
        msg = format_rejection_message(validation)
        assert msg is not None
        assert 'Uncertain' in msg or 'uncertain' in msg

    def test_valid_prediction_returns_none(self):
        validation = {'is_valid': True, 'rejection_type': None}
        msg = format_rejection_message(validation)
        assert msg is None


class TestValidationThresholds:
    """Tests for threshold configuration."""

    def test_thresholds_returned(self):
        thresholds = get_validation_thresholds()
        assert 'confidence_threshold' in thresholds
        assert 'margin_threshold' in thresholds
        assert 'not_animal_class' in thresholds

    def test_confidence_threshold_reasonable(self):
        thresholds = get_validation_thresholds()
        assert 0.3 <= thresholds['confidence_threshold'] <= 0.8


# ─── PREDICTION PIPELINE TESTS ──────────────────────────────────────────────

class TestPredictionPipeline:
    """Tests for the prediction pipeline (without trained model)."""

    def test_config_classes_count(self):
        assert NUM_CLASSES == len(CLASSES)

    def test_not_animal_class_in_classes(self):
        assert 'Not_Animal' in CLASSES

    def test_animal_classes_excludes_not_animal(self):
        assert 'Not_Animal' not in ANIMAL_CLASSES
        assert len(ANIMAL_CLASSES) == 15

    def test_img_size_is_224(self):
        assert IMG_SIZE == (224, 224)

    def test_all_animal_classes_have_emojis(self):
        from config import ANIMAL_EMOJIS
        for cls in ANIMAL_CLASSES:
            assert cls in ANIMAL_EMOJIS, f"Missing emoji for {cls}"


# ─── OOD DETECTOR TESTS ─────────────────────────────────────────────────────

class TestOODDetectorImports:
    """Test that OOD detector can be imported and has expected functions."""

    def test_import_ood_detector(self):
        from ood_detector import (
            build_feature_extractor, compute_ood_distance,
            mahalanobis_distance, extract_features
        )

    def test_mahalanobis_distance_zero_for_same_point(self):
        from ood_detector import mahalanobis_distance
        mean = np.zeros(10)
        cov_inv = np.eye(10)
        dist = mahalanobis_distance(mean, mean, cov_inv)
        assert dist == pytest.approx(0.0, abs=0.01)

    def test_mahalanobis_distance_positive_for_different(self):
        from ood_detector import mahalanobis_distance
        mean = np.zeros(10)
        point = np.ones(10)
        cov_inv = np.eye(10)
        dist = mahalanobis_distance(point, mean, cov_inv)
        assert dist > 0
