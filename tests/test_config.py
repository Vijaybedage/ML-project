"""
Unit tests for the shared configuration module.
"""

import os
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import (
    CLASSES, NUM_CLASSES, ANIMAL_EMOJIS, ANIMAL_FACTS,
    IMG_SIZE, BATCH_SIZE, SEED, MODEL_PATH, DATASET_PATH
)


class TestClasses:
    """Tests for class definitions."""

    def test_num_classes_matches_list(self):
        assert NUM_CLASSES == len(CLASSES)

    def test_exactly_16_classes(self):
        assert len(CLASSES) == 16

    def test_classes_are_sorted(self):
        assert CLASSES == sorted(CLASSES)

    def test_no_duplicate_classes(self):
        assert len(CLASSES) == len(set(CLASSES))

    def test_all_classes_have_emojis(self):
        for cls in CLASSES:
            assert cls in ANIMAL_EMOJIS, f"Missing emoji for {cls}"

    def test_all_classes_have_facts(self):
        for cls in CLASSES:
            assert cls in ANIMAL_FACTS, f"Missing fact for {cls}"

    def test_emoji_keys_match_classes(self):
        assert set(ANIMAL_EMOJIS.keys()) == set(CLASSES)

    def test_facts_keys_match_classes(self):
        assert set(ANIMAL_FACTS.keys()) == set(CLASSES)


class TestConfig:
    """Tests for configuration values."""

    def test_img_size_is_tuple(self):
        assert isinstance(IMG_SIZE, tuple)
        assert len(IMG_SIZE) == 2

    def test_img_size_values(self):
        assert IMG_SIZE == (224, 224)

    def test_batch_size_positive(self):
        assert BATCH_SIZE > 0

    def test_seed_is_set(self):
        assert isinstance(SEED, int)

    def test_model_path_has_extension(self):
        assert MODEL_PATH.endswith('.h5') or MODEL_PATH.endswith('.keras')

    def test_paths_are_absolute(self):
        assert os.path.isabs(MODEL_PATH)
        assert os.path.isabs(DATASET_PATH)


class TestExpectedClasses:
    """Verify the exact expected animal classes."""

    EXPECTED = {
        'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
        'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
        'Not_Animal', 'Panda', 'Tiger', 'Zebra'
    }

    def test_all_expected_classes_present(self):
        assert set(CLASSES) == self.EXPECTED
