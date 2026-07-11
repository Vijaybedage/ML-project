"""
Data Augmentation Pipeline Visualization
Shows all augmentations applied during training in a single grid.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.dirname(__file__))
from config import DATASET_PATH, RESULTS_PATH, IMG_SIZE, CLASSES, SEED

SAMPLE_IMAGE = os.path.join(DATASET_PATH, 'Lion', 'Lion_1.jpeg')
OUTPUT_PATH = os.path.join(RESULTS_PATH, 'augmentation_pipeline.png')


def visualize_augmentation_pipeline():
    """Create a grid showing original + each augmentation type."""
    if not os.path.exists(SAMPLE_IMAGE):
        # Find any available image
        for cls in CLASSES[:5]:
            cls_dir = os.path.join(DATASET_PATH, cls)
            if os.path.exists(cls_dir):
                imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imgs:
                    sample = os.path.join(cls_dir, imgs[0])
                    break
        else:
            print("No sample images found!")
            return
    else:
        sample = SAMPLE_IMAGE

    img = Image.open(sample).convert('RGB').resize(IMG_SIZE)
    img_arr = np.array(img) / 255.0
    img_batch = np.expand_dims(img_arr, 0)

    # Define individual augmentations
    augmentations = [
        ("Original", img_arr),
    ]

    # Rotation
    gen_rot = ImageDataGenerator(rotation_range=30)
    aug_rot = next(gen_rot.flow(img_batch, batch_size=1, seed=SEED))[0]
    augmentations.append(("Rotation\n(+/-30 deg)", aug_rot))

    # Width Shift
    gen_shift = ImageDataGenerator(width_shift_range=0.25)
    aug_shift = next(gen_shift.flow(img_batch, batch_size=1, seed=SEED))[0]
    augmentations.append(("Width Shift\n(+/-25%)", aug_shift))

    # Height Shift
    gen_hshift = ImageDataGenerator(height_shift_range=0.25)
    aug_hshift = next(gen_hshift.flow(img_batch, batch_size=1, seed=SEED))[0]
    augmentations.append(("Height Shift\n(+/-25%)", aug_hshift))

    # Shear
    gen_shear = ImageDataGenerator(shear_range=0.2)
    aug_shear = next(gen_shear.flow(img_batch, batch_size=1, seed=SEED))[0]
    augmentations.append(("Shear\n(0.2)", aug_shear))

    # Zoom
    gen_zoom = ImageDataGenerator(zoom_range=0.3)
    aug_zoom = next(gen_zoom.flow(img_batch, batch_size=1, seed=SEED))[0]
    augmentations.append(("Zoom\n(+/-30%)", aug_zoom))

    # Horizontal Flip
    gen_flip = ImageDataGenerator(horizontal_flip=True)
    aug_flip = next(gen_flip.flow(img_batch, batch_size=1, seed=SEED))[0]
    augmentations.append(("Horizontal\nFlip", aug_flip))

    # Brightness
    gen_bright = ImageDataGenerator(brightness_range=[0.8, 1.2])
    aug_bright = next(gen_bright.flow(img_batch, batch_size=1, seed=SEED))[0]
    augmentations.append(("Brightness\n(0.8-1.2)", aug_bright))

    # Combined (all augmentations together)
    gen_all = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect'
    )
    aug_all = next(gen_all.flow(img_batch, batch_size=1, seed=SEED))[0]
    augmentations.append(("Combined\n(All Together)", aug_all))

    # Create grid: 3 rows x 3 cols
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Data Augmentation Pipeline\nTraining Transformations',
                 fontsize=18, fontweight='bold', y=0.98)

    for idx, (title, aug_img) in enumerate(augmentations):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        ax.imshow(aug_img)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.axis('off')

        # Add colored border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#667eea' if idx == 0 else '#3498db')
            spine.set_linewidth(3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(RESULTS_PATH, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Augmentation pipeline saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    visualize_augmentation_pipeline()
