"""
Augment Bear class images to balance dataset.
Generates synthetic variations using geometric and color transforms.
Target: ~100 images (from 20 originals)
"""

import os
import sys
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging

sys.path.insert(0, os.path.dirname(__file__))
from config import DATASET_PATH, SEED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

TARGET_COUNT = 100
BEAR_DIR = os.path.join(DATASET_PATH, 'Bear')


def random_augment(img):
    """Apply random augmentation to a PIL image."""
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random rotation (-25 to 25 degrees)
    angle = random.uniform(-25, 25)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    # Random crop and resize (zoom effect)
    if random.random() > 0.5:
        w, h = img.size
        crop_pct = random.uniform(0.8, 0.95)
        new_w, new_h = int(w * crop_pct), int(h * crop_pct)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.BILINEAR)

    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))

    # Random contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))

    # Random saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.6, 1.4))

    # Random blur (mild)
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Random sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(random.uniform(0.5, 1.5))

    return img


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Get original images (those without numbers after underscore pattern)
    all_files = [f for f in os.listdir(BEAR_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Separate originals from augmented
    originals = [f for f in all_files if '_' not in f.split('.')[0].replace('Bear', '')]
    if len(originals) < 5:
        originals = all_files[:10]  # fallback

    logger.info(f"Found {len(originals)} original Bear images, {len(all_files)} total")

    existing = len(all_files)
    to_generate = TARGET_COUNT - existing

    if to_generate <= 0:
        logger.info(f"Bear class already has {existing} images (target: {TARGET_COUNT}). Skipping.")
        return

    logger.info(f"Generating {to_generate} new Bear images to reach {TARGET_COUNT}...")

    generated = 0
    while generated < to_generate:
        src_file = random.choice(originals)
        src_path = os.path.join(BEAR_DIR, src_file)
        img = Image.open(src_path).convert('RGB')
        img = img.resize((224, 224))

        augmented = random_augment(img)

        new_name = f"Bear_aug_{existing + generated + 1:03d}.jpg"
        new_path = os.path.join(BEAR_DIR, new_name)
        augmented.save(new_path, 'JPEG', quality=90)

        generated += 1
        if generated % 10 == 0:
            logger.info(f"  Generated {generated}/{to_generate}...")

    final_count = len([f for f in os.listdir(BEAR_DIR)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    logger.info(f"Done! Bear class now has {final_count} images.")


if __name__ == '__main__':
    main()
