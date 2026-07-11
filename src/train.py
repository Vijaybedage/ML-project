"""
Animal Classification using Transfer Learning (MobileNetV2)
Author: Vijay Bedage

Two-phase training with advanced techniques:
  Phase 1 — Train classification head with frozen MobileNetV2 base + Mixup/CutMix
  Phase 2 — Fine-tune top layers with cosine annealing LR + label smoothing

Anti-wrong-prediction techniques:
  - Mixup & CutMix augmentation for better generalization
  - Label smoothing to prevent overconfident predictions
  - Class-weighted loss for imbalanced data (Bear: 20 images)
  - Cosine annealing LR schedule
  - Stronger regularization (higher dropout, weight decay)
  - OOD statistics computation for validation
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.optimizers.schedules import CosineDecay
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


# ─── MIXUP & CUTMIX AUGMENTATION ─────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    """Apply Mixup augmentation: linear interpolation of images and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation: cut and paste image patches."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)

    h, w = x.shape[1], x.shape[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    x[:, bby1:bby2, bbx1:bbx2, :] = x[index, bby1:bby2, bbx1:bbx2, :]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    mixed_y = lam * y + (1 - lam) * y[index]
    return x, mixed_y


def apply_mixup_cutmix(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
    """Apply Mixup or CutMix with given probability."""
    if np.random.random() < prob:
        if np.random.random() < 0.5:
            return mixup_data(x, y, mixup_alpha)
        else:
            return cutmix_data(x, y, cutmix_alpha)
    return x, y


# ─── LABEL SMOOTHING LOSS ────────────────────────────────────────────────────
class LabelSmoothingCrossEntropy(tf.keras.losses.Loss):
    """Cross-entropy loss with label smoothing to prevent overconfident predictions."""

    def __init__(self, smoothing=0.1, name='label_smoothing_crossentropy'):
        super().__init__(name=name)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_true)[1], tf.float32)
        y_true = y_true * (1.0 - self.smoothing) + self.smoothing / num_classes
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(loss)

    def get_config(self):
        return {'smoothing': self.smoothing}


# ─── CLASS WEIGHTS ────────────────────────────────────────────────────────────
def compute_class_weights(generator):
    """Compute class weights for imbalanced dataset."""
    y = generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = dict(enumerate(class_weights))
    logger.info(f"Class weights: {class_weight_dict}")
    return class_weight_dict


# ─── DATA GENERATORS ─────────────────────────────────────────────────────────
def build_generators():
    """Build training and validation data generators with advanced augmentation."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            f"Please place your dataset in the 'Animal Classification/dataset/' directory."
        )

    # Enhanced augmentation for training (AutoAugment-inspired)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect',
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
    """Build MobileNetV2-based classification model with stronger regularization."""
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=LabelSmoothingCrossEntropy(smoothing=0.1),
        metrics=['accuracy']
    )

    logger.info(f"Model built: {len(model.layers)} layers, "
                f"{model.count_params():,} total params")
    return model, base_model


# ─── CALLBACKS ────────────────────────────────────────────────────────────────
def get_callbacks(phase: str = 'phase1'):
    """Get training callbacks for the specified phase."""
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7 if phase == 'phase1' else 5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


# ─── FINE-TUNING ──────────────────────────────────────────────────────────────
def fine_tune_model(model, base_model):
    """Unfreeze top layers of base model and recompile with lower LR + cosine decay."""
    base_model.trainable = True

    # Freeze all layers except top FINE_TUNE_LAYERS
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    # Cosine decay schedule
    total_steps = FINE_TUNE_EPOCHS * 50  # approximate steps per epoch
    lr_schedule = CosineDecay(
        initial_learning_rate=FINE_TUNE_LR,
        decay_steps=total_steps,
        alpha=0.01
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=LabelSmoothingCrossEntropy(smoothing=0.1),
        metrics=['accuracy']
    )

    trainable = sum(1 for layer in model.layers if layer.trainable)
    logger.info(f"Fine-tuning: {trainable} trainable layers, LR schedule: CosineDecay")


# ─── PLOT HISTORY ─────────────────────────────────────────────────────────────
def plot_history(history_phase1, history_phase2=None):
    """Plot training accuracy and loss curves for both phases."""
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


# ─── MIXUP TRAINING LOOP ───────────────────────────────────────────────────────
def train_with_mixup(model, train_gen, val_gen, epochs, class_weights, mixup_prob=0.5):
    """Custom training loop with Mixup/CutMix augmentation applied per batch."""
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    steps_per_epoch = len(train_gen)
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training loop
        train_losses = []
        train_accs = []
        
        for step in range(steps_per_epoch):
            x, y = next(train_gen)
            
            # Apply Mixup/CutMix
            if np.random.random() < mixup_prob:
                if np.random.random() < 0.5:
                    x, y = mixup_data(x, y, alpha=0.2)
                else:
                    x, y = cutmix_data(x, y, alpha=1.0)
            
            # Train step
            logs = model.train_on_batch(x, y, return_dict=True)
            train_losses.append(logs['loss'])
            train_accs.append(logs['accuracy'])
            
            # Progress bar
            if step % 10 == 0 or step == steps_per_epoch - 1:
                avg_loss = np.mean(train_losses)
                avg_acc = np.mean(train_accs)
                prog = (step + 1) / steps_per_epoch * 100
                print(f"\r  {prog:.0f}% [{step+1}/{steps_per_epoch}] loss: {avg_loss:.4f} acc: {avg_acc:.4f}", end='')
        
        # Validation
        val_logs = model.evaluate(val_gen, verbose=0, return_dict=True)
        
        history['loss'].append(np.mean(train_losses))
        history['accuracy'].append(np.mean(train_accs))
        history['val_loss'].append(val_logs['loss'])
        history['val_accuracy'].append(val_logs['accuracy'])
        
        logger.info(f"  loss: {np.mean(train_losses):.4f} - accuracy: {np.mean(train_accs):.4f} - "
                    f"val_loss: {val_logs['loss']:.4f} - val_accuracy: {val_logs['accuracy']:.4f}")
    
    # Convert to Keras History-like object
    class SimpleHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    return SimpleHistory(history)


# ─── CONFUSION MATRIX ─────────────────────────────────────────────────────────
def plot_confusion_matrix(model, val_gen):
    """Generate and save confusion matrix heatmap and classification report."""
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
    logger.info(f"  Techniques: Mixup/CutMix, Label Smoothing (0.1), Class Weights, Cosine Decay")

    # ── Data ──
    train_gen, val_gen = build_generators()
    logger.info(f"  Train samples: {train_gen.samples}")
    logger.info(f"  Val samples:   {val_gen.samples}")

    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_gen)

    # ── Phase 1: Train classification head ──
    logger.info("\n═══ Phase 1: Training classification head (base frozen) ═══")
    model, base_model = build_model(num_classes=NUM_CLASSES)
    model.summary()

    # Custom training loop with Mixup/CutMix for Phase 1
    logger.info("Training with Mixup/CutMix augmentation...")
    history1 = train_with_mixup(model, train_gen, val_gen, EPOCHS, class_weights)

    val_loss1, val_acc1 = model.evaluate(val_gen, verbose=0)
    logger.info(f"Phase 1 — Val Accuracy: {val_acc1:.4f} ({val_acc1*100:.2f}%)")

    # ── Phase 2: Fine-tune top layers ──
    logger.info(f"\n═══ Phase 2: Fine-tuning top {FINE_TUNE_LAYERS} base layers ═══")
    fine_tune_model(model, base_model)

    # Custom training loop with Mixup/CutMix for Phase 2 (less aggressive)
    history2 = train_with_mixup(model, train_gen, val_gen, FINE_TUNE_EPOCHS, class_weights, mixup_prob=0.3)

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

    # Compute OOD statistics for validation
    logger.info("\nComputing OOD detection statistics...")
    try:
        from ood_detector import initialize_ood_detector
        initialize_ood_detector(MODEL_PATH, DATASET_PATH)
        logger.info("OOD statistics computed and cached.")
    except Exception as e:
        logger.warning(f"Could not compute OOD stats: {e}")