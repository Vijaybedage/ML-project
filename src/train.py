"""
Animal Classification using Transfer Learning (MobileNetV2)
Author: Vijay Bedage

Two-phase training:
  Phase 1 — Train classification head with frozen MobileNetV2 base
  Phase 2 — Fine-tune top layers of the base model with a lower learning rate
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
import logging

# Add parent directory for config import when running as script
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATASET_PATH, MODEL_PATH, RESULTS_PATH, IMG_SIZE, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, FINE_TUNE_EPOCHS, FINE_TUNE_LR,
    FINE_TUNE_LAYERS, SEED, CLASSES, NUM_CLASSES
)

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─── REPRODUCIBILITY ─────────────────────────────────────────────────────────
def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


# ─── DATA GENERATORS ─────────────────────────────────────────────────────────
def build_generators():
    """Build training and validation data generators with augmentation.

    Training generator applies random augmentations (rotation, shift, zoom, flip)
    to improve model generalization. Validation generator only rescales.

    Returns:
        tuple: (train_generator, validation_generator)
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            f"Please place your dataset in the 'Animal Classification/dataset/' directory."
        )

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=SEED,
        classes=CLASSES
    )

    val_gen = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=SEED,
        classes=CLASSES
    )

    return train_gen, val_gen


# ─── MODEL ────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES):
    """Build MobileNetV2-based classification model.

    Architecture:
        MobileNetV2 (frozen) → GlobalAvgPool → BN → Dense(256) → Dropout(0.5)
        → Dense(128) → Dropout(0.3) → Dense(num_classes, softmax)

    Args:
        num_classes: Number of output classes.

    Returns:
        Compiled Keras Model.
    """
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze base model initially (Phase 1)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info(f"Model built: {len(model.layers)} layers, "
                f"{model.count_params():,} total params")
    return model, base_model


# ─── CALLBACKS ────────────────────────────────────────────────────────────────
def get_callbacks(phase: str = 'phase1'):
    """Get training callbacks for the specified phase.

    Args:
        phase: 'phase1' (head training) or 'phase2' (fine-tuning).

    Returns:
        List of Keras callbacks.
    """
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7 if phase == 'phase1' else 5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]


# ─── FINE-TUNING ──────────────────────────────────────────────────────────────
def fine_tune_model(model, base_model):
    """Unfreeze the top layers of the base model and recompile with a lower LR.

    This is Phase 2 of training — fine-tuning the top layers of MobileNetV2
    for domain-specific feature adaptation.

    Args:
        model: The full model (base + classification head).
        base_model: The MobileNetV2 base model.
    """
    base_model.trainable = True

    # Freeze all layers except the top FINE_TUNE_LAYERS
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    trainable = sum(1 for layer in model.layers if layer.trainable)
    logger.info(f"Fine-tuning: {trainable} trainable layers, LR={FINE_TUNE_LR}")


# ─── PLOT HISTORY ─────────────────────────────────────────────────────────────
def plot_history(history_phase1, history_phase2=None):
    """Plot training accuracy and loss curves for both phases.

    Args:
        history_phase1: Keras History object from Phase 1.
        history_phase2: Optional Keras History object from Phase 2.
    """
    # Combine histories
    acc = history_phase1.history['accuracy']
    val_acc = history_phase1.history['val_accuracy']
    loss = history_phase1.history['loss']
    val_loss = history_phase1.history['val_loss']

    phase1_epochs = len(acc)

    if history_phase2:
        acc += history_phase2.history['accuracy']
        val_acc += history_phase2.history['val_accuracy']
        loss += history_phase2.history['loss']
        val_loss += history_phase2.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, acc, label='Train Accuracy', color='royalblue')
    axes[0].plot(epochs_range, val_acc, label='Val Accuracy', color='coral')
    if history_phase2:
        axes[0].axvline(x=phase1_epochs, color='gray', linestyle='--',
                        alpha=0.5, label='Fine-tuning start')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, loss, label='Train Loss', color='royalblue')
    axes[1].plot(epochs_range, val_loss, label='Val Loss', color='coral')
    if history_phase2:
        axes[1].axvline(x=phase1_epochs, color='gray', linestyle='--',
                        alpha=0.5, label='Fine-tuning start')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'training_history.png'), dpi=150)
    plt.close()
    logger.info("Training history plot saved.")


# ─── CONFUSION MATRIX ─────────────────────────────────────────────────────────
def plot_confusion_matrix(model, val_gen):
    """Generate and save a confusion matrix heatmap and classification report.

    Args:
        model: Trained Keras model.
        val_gen: Validation data generator.
    """
    val_gen.reset()
    preds = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'), dpi=150)
    plt.close()
    logger.info("Confusion matrix saved.")

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    logger.info(f"\nClassification Report:\n{report}")

    with open(os.path.join(RESULTS_PATH, 'classification_report.txt'), 'w') as f:
        f.write(report)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(SEED)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    logger.info("Starting Animal Classification Training...")
    logger.info(f"  Classes: {NUM_CLASSES}")
    logger.info(f"  Image Size: {IMG_SIZE}")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  Seed: {SEED}")

    # ── Data ──
    train_gen, val_gen = build_generators()
    logger.info(f"  Train samples: {train_gen.samples}")
    logger.info(f"  Val samples:   {val_gen.samples}")

    # ── Phase 1: Train classification head ──
    logger.info("\n═══ Phase 1: Training classification head (base frozen) ═══")
    model, base_model = build_model(num_classes=NUM_CLASSES)
    model.summary()

    history1 = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks('phase1'),
        verbose=1
    )

    val_loss1, val_acc1 = model.evaluate(val_gen, verbose=0)
    logger.info(f"Phase 1 — Val Accuracy: {val_acc1:.4f} ({val_acc1*100:.2f}%)")

    # ── Phase 2: Fine-tune top layers ──
    logger.info(f"\n═══ Phase 2: Fine-tuning top {FINE_TUNE_LAYERS} base layers ═══")
    fine_tune_model(model, base_model)

    history2 = model.fit(
        train_gen,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks('phase2'),
        verbose=1
    )

    # ── Results ──
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    logger.info(f"\nFinal Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    logger.info(f"Improvement from fine-tuning: {(val_acc - val_acc1)*100:+.2f}%")
    logger.info(f"Model saved to: {MODEL_PATH}")

    # Save combined training history
    hist_combined = {}
    for key in history1.history:
        hist_combined[key] = (
            [float(v) for v in history1.history[key]] +
            [float(v) for v in history2.history[key]]
        )
    with open(os.path.join(RESULTS_PATH, 'history.json'), 'w') as f:
        json.dump(hist_combined, f, indent=2)

    plot_history(history1, history2)
    plot_confusion_matrix(model, val_gen)
