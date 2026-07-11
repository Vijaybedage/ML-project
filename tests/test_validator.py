"""
Unit tests for the 3-tier prediction validation system.
"""

import os
import sys
import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from validator import (
    validate_prediction, format_rejection_message, compute_entropy,
    temperature_scale, ANIMAL_CLASSES
)
from config import CLASSES


class TestTier1AutoAccept:
    """Tier 1: High confidence predictions should be auto-accepted."""

    def test_lion_100_percent_accepted(self):
        """A lion at 100% confidence must NEVER be rejected."""
        probs = np.zeros(16)
        probs[CLASSES.index('Lion')] = 1.0
        result = validate_prediction(probs)
        assert result['is_valid'] is True
        assert result['tier'] == 1
        assert result['rejection_type'] is None

    def test_high_confidence_auto_accept(self):
        """95% confidence with clear margin → auto-accept."""
        probs = np.array([0.95, 0.03, 0.02] + [0.0] * 13)
        result = validate_prediction(probs)
        assert result['is_valid'] is True
        assert result['tier'] == 1

    def test_threshold_boundary_accept(self):
        """Exactly at the high confidence threshold → should accept."""
        probs = np.zeros(16)
        probs[0] = 0.92  # Well above CONFIDENCE_THRESHOLD after temp scaling
        probs[1] = 0.05
        probs[2] = 0.03
        result = validate_prediction(probs)
        assert result['is_valid'] is True


class TestTier2AutoReject:
    """Tier 2: Very low confidence should be auto-rejected as 'uncertain'."""

    def test_uniform_distribution_rejected(self):
        """Near-uniform probabilities → rejected as uncertain."""
        probs = np.ones(16) / 16
        result = validate_prediction(probs)
        assert result['is_valid'] is False
        assert result['rejection_type'] == 'uncertain'
        assert result['tier'] == 3

    def test_very_low_confidence_rejected(self):
        """Very low top confidence → rejected as uncertain."""
        probs = np.array([0.15, 0.12, 0.10, 0.09, 0.08] + [0.046] * 11)
        result = validate_prediction(probs)
        assert result['is_valid'] is False
        assert result['rejection_type'] == 'uncertain'

    def test_not_animal_no_prediction_shown(self):
        """Rejection message for 'not_animal' should NOT show predictions."""
        probs = np.zeros(16)
        probs[CLASSES.index('Not_Animal')] = 0.90
        probs[CLASSES.index('Lion')] = 0.10
        result = validate_prediction(probs)
        assert result['rejection_type'] == 'not_animal'
        msg = format_rejection_message(result)
        assert 'Not a Recognized Animal' in msg
        assert 'Top prediction' not in msg  # Should NOT show predictions


class TestTier3ModerateZone:
    """Tier 3: Moderate confidence — multi-criteria check."""

    def test_moderate_confidence_single_check_passes(self):
        """Confidence above threshold should be accepted."""
        probs = np.array([0.70, 0.15, 0.10, 0.05] + [0.0] * 12)
        result = validate_prediction(probs)
        assert result['is_valid'] is True
        assert result['tier'] == 1

    def test_uncertain_shows_possibilities(self):
        """'uncertain' rejection should list supported animals."""
        probs = np.array([0.45, 0.35, 0.10, 0.05, 0.05] + [0.0] * 11)
        result = validate_prediction(probs)
        assert result['is_valid'] is False
        assert result['rejection_type'] == 'uncertain'
        msg = format_rejection_message(result)
        assert 'Uncertain' in msg
        assert 'Supported animals' in msg


class TestRejectionMessages:
    """Test formatted rejection messages."""

    def test_not_animal_message_lists_supported(self):
        """Not-animal message should list supported animals."""
        probs = np.zeros(16)
        probs[CLASSES.index('Not_Animal')] = 0.90
        result = validate_prediction(probs)
        msg = format_rejection_message(result)
        for animal in ANIMAL_CLASSES:
            assert animal in msg

    def test_valid_prediction_no_message(self):
        """Valid predictions should return None for rejection message."""
        probs = np.zeros(16)
        probs[0] = 0.98
        probs[1] = 0.02
        result = validate_prediction(probs)
        msg = format_rejection_message(result)
        assert msg is None


class TestHelperFunctions:
    """Test entropy and temperature scaling."""

    def test_entropy_uniform(self):
        """Uniform distribution should have maximum entropy."""
        probs = np.ones(16) / 16
        ent = compute_entropy(probs)
        assert ent > 3.5  # log2(16) = 4.0

    def test_entropy_certain(self):
        """Certain prediction should have near-zero entropy."""
        probs = np.zeros(16)
        probs[0] = 1.0
        ent = compute_entropy(probs)
        assert ent < 0.01

    def test_temperature_scaling_sums_to_one(self):
        """Temperature-scaled probabilities should sum to 1."""
        logits = np.random.randn(16)
        scaled = temperature_scale(logits, temperature=1.5)
        assert abs(np.sum(scaled) - 1.0) < 1e-6

    def test_higher_temperature_softer(self):
        """Higher temperature should produce softer (more uniform) distribution."""
        logits = np.array([5.0, 1.0, 0.5] + [0.0] * 13)
        soft = temperature_scale(logits, temperature=3.0)
        hard = temperature_scale(logits, temperature=0.5)
        # Soft should have higher entropy (more uniform)
        assert compute_entropy(soft) > compute_entropy(hard)


class TestResultStructure:
    """Ensure validation results have all required fields."""

    def test_result_has_required_fields(self):
        probs = np.zeros(16)
        probs[0] = 0.9
        probs[1] = 0.1
        result = validate_prediction(probs)
        required = [
            'is_valid', 'rejection_type', 'reason', 'tier',
            'confidence', 'margin', 'entropy', 'ood_distance',
            'top1_class', 'top2_class', 'calibrated_probs'
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_top1_class_is_valid(self):
        probs = np.zeros(16)
        probs[5] = 0.95
        result = validate_prediction(probs)
        assert result['top1_class'] in CLASSES
