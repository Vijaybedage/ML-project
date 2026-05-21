"""
Animal Classification using Transfer Learning (MobileNetV2)
Author: Vijay Bedage
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'Animal Classification', 'dataset')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'animal_classifier.h5')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

CLASSES = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
           'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
           'Panda', 'Tiger', 'Zebra']

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


# ─── DATA GENERATORS ──────────────────────────────────────────────────────────
def build_generators():
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
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen


# ─── MODEL ────────────────────────────────────────────────────────────────────
def build_model(num_classes=15):
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze base model initially
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
    return model


# ─── CALLBACKS ────────────────────────────────────────────────────────────────
def get_callbacks():
    return [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]


# ─── PLOT HISTORY ─────────────────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='royalblue')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='coral')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train Loss', color='royalblue')
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='coral')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'training_history.png'), dpi=150)
    plt.close()
    print("✅ Training history plot saved.")


# ─── CONFUSION MATRIX ─────────────────────────────────────────────────────────
def plot_confusion_matrix(model, val_gen):
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
    print("✅ Confusion matrix saved.")

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print("\n📊 Classification Report:\n", report)

    with open(os.path.join(RESULTS_PATH, 'classification_report.txt'), 'w') as f:
        f.write(report)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("🚀 Starting Animal Classification Training...")
    print(f"   Classes: {len(CLASSES)}")
    print(f"   Image Size: {IMG_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")

    train_gen, val_gen = build_generators()
    print(f"   Train samples: {train_gen.samples}")
    print(f"   Val samples:   {val_gen.samples}")

    model = build_model(num_classes=len(CLASSES))
    model.summary()

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Save training history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(RESULTS_PATH, 'history.json'), 'w') as f:
        json.dump(hist_dict, f, indent=2)

    plot_history(history)
    plot_confusion_matrix(model, val_gen)

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n🎯 Final Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
