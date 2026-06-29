# src/step2_process_custom_images.py
# Extracts 21 hand landmarks from static sign drawings using contour-moments.
# Balances dataset to 500 samples per class (skips classes with fewer than 20 PNGs).

import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# Minimum PNG images needed before a class is included in training
MIN_CLASS_IMAGES = 20
# Target samples per class (achieved via up-sampling if needed)
TARGET_PER_CLASS = 150

def get_image_tuples(data_dir):
    image_tuples = []
    # Discover all valid classes above the minimum threshold
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        png_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.png')]
        if len(png_files) < MIN_CLASS_IMAGES:
            print(f"  Skipping class '{class_name}': only {len(png_files)} PNGs (need {MIN_CLASS_IMAGES})")
            continue
        # Up-sample to TARGET_PER_CLASS by repeating available images
        paths = [os.path.join(class_dir, f) for f in png_files]
        if len(paths) >= TARGET_PER_CLASS:
            selected = paths[:TARGET_PER_CLASS]
        else:
            # Repeat images until we reach target
            repeats = (TARGET_PER_CLASS // len(paths)) + 1
            selected = (paths * repeats)[:TARGET_PER_CLASS]
        for p in selected:
            image_tuples.append((p, class_name))
    return image_tuples

def extract_wrist(img):
    # Red landmarks = wrist (BGR: 48, 48, 255)
    mask = cv2.inRange(img, np.array([48, 48, 255]), np.array([48, 48, 255]))
    y_red, x_red = np.where(mask > 0)
    if len(x_red) == 0:
        return np.array([0.0, 0.0])
    return np.array([np.mean(x_red), np.mean(y_red)])

def extract_finger_joints(img, color, wrist):
    mask = cv2.inRange(img, np.array(color), np.array(color))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for c in contours:
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            centroids.append((cx, cy, cv2.contourArea(c)))
    # Keep 4 largest joint circles, sort by distance to wrist
    centroids = [c for c in centroids if c[2] > 2]
    centroids = sorted(centroids, key=lambda x: -x[2])[:4]
    centroids = sorted([np.array([c[0], c[1]]) for c in centroids],
                       key=lambda p: np.linalg.norm(p - wrist))
    while len(centroids) < 4:
        centroids.append(centroids[-1] if centroids else wrist)
    return centroids

def extract_hand_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    wrist = extract_wrist(img)
    colors_map = {
        'Thumb':  [180, 229, 255],
        'Index':  [192, 101,  21],
        'Middle': [ 48, 255,  48],
        'Ring':   [  0, 204, 255],
        'Pinky':  [128,  64, 128],
    }
    landmarks = [wrist]
    for finger, color in colors_map.items():
        landmarks.extend(extract_finger_joints(img, color, wrist))
    # Normalise by image width/height (400 px) and add Z=0
    lms = np.array(landmarks, dtype=np.float32) / 400.0
    return np.hstack([lms, np.zeros((21, 1), dtype=np.float32)])

def build_sequence(hand_xyz):
    # Tile single frame 30 times; pad 45 zeros for empty pose context → (30, 108)
    seq_hand = np.tile(hand_xyz.flatten(), (30, 1))
    seq_pose = np.zeros((30, 45), dtype=np.float32)
    return np.concatenate([seq_hand, seq_pose], axis=1)

def process_dataset(image_tuples, label_map):
    X, y = [], []
    for img_path, label in tqdm(image_tuples, desc="Extracting landmarks"):
        hand_xyz = extract_hand_landmarks(img_path)
        if hand_xyz is not None:
            X.append(build_sequence(hand_xyz))
            y.append(label_map[label])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def save_data(X, y, label_map, dest_dir='data/processed'):
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
    np.save(os.path.join(dest_dir, 'X_train.npy'), X)
    np.save(os.path.join(dest_dir, 'y_train.npy'), y)
    print(f"\nSaved -> X={X.shape}, y={y.shape}")
    print(f"Classes: {list(label_map.keys())}")

if __name__ == '__main__':
    data_dir = 'dataset/processed_combine_asl_dataset'
    print(f"Building dataset (min {MIN_CLASS_IMAGES} images, target {TARGET_PER_CLASS}/class)...")
    image_tuples = get_image_tuples(data_dir)
    labels = sorted(set(lbl for _, lbl in image_tuples))
    label_map = {lbl: idx for idx, lbl in enumerate(labels)}
    print(f"Found {len(image_tuples)} total samples across {len(labels)} classes: {labels}")
    X, y = process_dataset(image_tuples, label_map)
    save_data(X, y, label_map)
