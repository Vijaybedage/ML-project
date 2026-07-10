"""
OOD (Out-of-Distribution) Detection for Animal Classification
Author: Vijay Bedage

Uses feature extractor from trained model to compute Mahalanobis distance
or other distance metrics to detect out-of-distribution inputs.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import logging

# Add parent directory for config import
sys.path.insert(0, os.path.dirname(__file__))
from config import MODEL_PATH, IMG_SIZE, CLASSES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Global feature extractor and statistics
_FEATURE_EXTRACTOR = None
_CLASS_MEANS = None
_CLASS_COV_INV = None
_GLOBAL_MEAN = None
_GLOBAL_COV_INV = None


def build_feature_extractor(model_path: str = MODEL_PATH):
    """
    Build a feature extractor model from the trained classifier.
    
    Extracts features from the GlobalAveragePooling2D layer (penultimate layer
    before classification head).
    
    Args:
        model_path: Path to the trained Keras model.
        
    Returns:
        Keras Model that outputs features instead of class probabilities.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    # Find the GlobalAveragePooling2D layer (feature layer before classification head)
    feature_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            feature_layer = layer
            break
    
    if feature_layer is None:
        # Fallback: use the layer before the final Dense layer
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Dense) and layer.units == len(CLASSES):
                # The layer before this should be our feature layer
                if i > 0:
                    feature_layer = model.layers[i - 1]
                break
    
    if feature_layer is None:
        raise ValueError("Could not find feature extraction layer in model")
    
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=feature_layer.output,
        name='feature_extractor'
    )
    
    logger.info(f"Built feature extractor from layer: {feature_layer.name}")
    return feature_extractor


def get_feature_extractor(model_path: str = MODEL_PATH):
    """Get or build the feature extractor (cached)."""
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        _FEATURE_EXTRACTOR = build_feature_extractor(model_path)
    return _FEATURE_EXTRACTOR


def compute_class_statistics(feature_extractor, dataset_path, batch_size=32):
    """
    Compute per-class mean and covariance of features from training data.
    
    Args:
        feature_extractor: Keras model that outputs features.
        dataset_path: Path to training dataset directory.
        batch_size: Batch size for processing.
        
    Returns:
        Tuple of (class_means, class_cov_inv, global_mean, global_cov_inv)
    """
    from config import DATASET_PATH, BATCH_SIZE, SEED
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=False,
        seed=SEED,
        classes=CLASSES
    )
    
    logger.info(f"Computing statistics on {train_gen.samples} training samples...")
    
    all_features = []
    all_labels = []
    
    for i in range(len(train_gen)):
        batch_x, batch_y = train_gen[i]
        features = feature_extractor.predict(batch_x, verbose=0)
        all_features.append(features)
        all_labels.append(batch_y)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {(i + 1) * BATCH_SIZE} samples...")
    
    all_features = np.vstack(all_features)
    all_labels = np.vstack(all_labels)
    y_indices = np.argmax(all_labels, axis=1)
    
    # Compute per-class statistics
    class_means = {}
    class_cov_inv = {}
    
    for class_idx in range(len(CLASSES)):
        class_features = all_features[y_indices == class_idx]
        if len(class_features) > 1:
            mean = np.mean(class_features, axis=0)
            cov = np.cov(class_features, rowvar=False)
            # Add regularization for numerical stability
            cov_reg = cov + 1e-4 * np.eye(cov.shape[0])
            try:
                cov_inv = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_reg)
        else:
            mean = class_features[0] if len(class_features) > 0 else np.zeros(all_features.shape[1])
            cov_inv = np.eye(all_features.shape[1])
        
        class_means[class_idx] = mean
        class_cov_inv[class_idx] = cov_inv
    
    # Global statistics
    global_mean = np.mean(all_features, axis=0)
    global_cov = np.cov(all_features, rowvar=False)
    global_cov_reg = global_cov + 1e-4 * np.eye(global_cov.shape[0])
    try:
        global_cov_inv = np.linalg.inv(global_cov_reg)
    except np.linalg.LinAlgError:
        global_cov_inv = np.linalg.pinv(global_cov_reg)
    
    logger.info("Statistics computation complete.")
    
    return class_means, class_cov_inv, global_mean, global_cov_inv


def initialize_ood_detector(model_path: str = MODEL_PATH, dataset_path: str = None):
    """
    Initialize OOD detector by building feature extractor and computing statistics.
    
    Args:
        model_path: Path to trained model.
        dataset_path: Path to training dataset (uses config default if None).
    """
    global _CLASS_MEANS, _CLASS_COV_INV, _GLOBAL_MEAN, _GLOBAL_COV_INV
    
    logger.info("Initializing OOD detector...")
    
    feature_extractor = get_feature_extractor(model_path)
    
    if dataset_path is None:
        from config import DATASET_PATH
        dataset_path = DATASET_PATH
    
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset not found at {dataset_path}. OOD detection will use global stats only.")
        _CLASS_MEANS = {}
        _CLASS_COV_INV = {}
        _GLOBAL_MEAN = np.zeros(feature_extractor.output_shape[-1])
        _GLOBAL_COV_INV = np.eye(feature_extractor.output_shape[-1])
        return
    
    _CLASS_MEANS, _CLASS_COV_INV, _GLOBAL_MEAN, _GLOBAL_COV_INV = \
        compute_class_statistics(feature_extractor, dataset_path)
    
    logger.info("OOD detector initialized.")


def mahalanobis_distance(x, mean, cov_inv):
    """Compute Mahalanobis distance between x and mean with inverse covariance."""
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff))


def compute_ood_distance(features):
    """
    Compute OOD distance for a feature vector.
    
    Uses minimum Mahalanobis distance to any class distribution.
    
    Args:
        features: Feature vector of shape (feature_dim,) or (1, feature_dim).
        
    Returns:
        OOD distance (lower = more in-distribution).
    """
    global _CLASS_MEANS, _CLASS_COV_INV, _GLOBAL_MEAN, _GLOBAL_COV_INV
    
    if features.ndim == 2:
        features = features[0]
    
    # Initialize if not done
    if _CLASS_MEANS is None:
        initialize_ood_detector()
    
    if _CLASS_MEANS and _CLASS_COV_INV:
        # Use per-class distances, take minimum
        distances = []
        for class_idx in _CLASS_MEANS:
            dist = mahalanobis_distance(features, _CLASS_MEANS[class_idx], _CLASS_COV_INV[class_idx])
            distances.append(dist)
        return min(distances)
    else:
        # Fallback to global distance
        return mahalanobis_distance(features, _GLOBAL_MEAN, _GLOBAL_COV_INV)


def extract_features(image_array):
    """
    Extract features from a preprocessed image array.
    
    Args:
        image_array: Preprocessed image of shape (1, 224, 224, 3) with values in [0, 1].
        
    Returns:
        Feature vector of shape (feature_dim,).
    """
    feature_extractor = get_feature_extractor()
    features = feature_extractor.predict(image_array, verbose=0)
    return features[0]


if __name__ == '__main__':
    # Test feature extractor build
    try:
        fe = build_feature_extractor()
        print(f"Feature extractor built successfully. Output shape: {fe.output_shape}")
    except Exception as e:
        print(f"Error building feature extractor: {e}")