# src/step2_process_custom_images.py
# This script extracts 21 hand landmarks from static drawing images
# using color-isolated K-Means and structures them into sequence data.

import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

def get_image_tuples(data_dir, max_per_class=200):
    image_tuples = []
    # Loop over class subdirectories (0-9, a-z)
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            count = 0
            for filename in os.listdir(class_dir):
                if filename.lower().endswith('.png'):
                    # Store path and label
                    image_tuples.append((os.path.join(class_dir, filename), class_name))
                    count += 1
                    if count >= max_per_class:
                        break
    return image_tuples

def extract_wrist(img):
    mask = cv2.inRange(img, np.array([48, 48, 255]), np.array([48, 48, 255]))
    y_red, x_red = np.where(mask > 0)
    if len(x_red) == 0:
        return np.array([0.0, 0.0])
    # Centroid of red wrist pixels
    return np.array([np.mean(x_red), np.mean(y_red)])

def extract_finger_joints(img, color):
    mask = cv2.inRange(img, np.array(color), np.array(color))
    y, x = np.where(mask > 0)
    if len(x) < 4:
        return [np.array([0.0, 0.0])] * 4
    pixels = np.column_stack((x, y))
    # Cluster index line pixels into 4 joint centers
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=3)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_

def extract_hand_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    wrist = extract_wrist(img)
    colors_map = {
        'Thumb': [180, 229, 255], 'Index': [192, 101, 21],
        'Middle': [48, 255, 48], 'Ring': [0, 204, 255], 'Pinky': [128, 64, 128]
    }
    sorted_landmarks = [wrist]
    # Process each finger and sort by distance to wrist
    for finger, color in colors_map.items():
        joints = extract_finger_joints(img, color)
        sorted_landmarks.extend(sorted(joints, key=lambda p: np.linalg.norm(p - wrist)))
    # Add Z=0.0 and normalize coordinates by image size (400x400)
    lms = np.array(sorted_landmarks, dtype=np.float32) / 400.0
    return np.hstack([lms, np.zeros((21, 1), dtype=np.float32)])

def build_sequence(hand_xyz):
    # Repeat the 63-feature hand coordinates 30 times
    sequence_hand = np.tile(hand_xyz.flatten(), (30, 1))
    # Pad with 45 zeros representing empty pose context (63 + 45 = 108 features)
    sequence_pose = np.zeros((30, 45), dtype=np.float32)
    return np.concatenate([sequence_hand, sequence_pose], axis=1)

def process_custom_dataset(image_tuples, label_map):
    X, y = [], []
    for img_path, label in tqdm(image_tuples, desc="Processing images"):
        hand_xyz = extract_hand_landmarks(img_path)
        if hand_xyz is not None:
            seq = build_sequence(hand_xyz)
            X.append(seq)
            y.append(label_map[label])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def save_custom_processed_data(X, y, label_map, dest_dir='data/processed'):
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
    np.save(os.path.join(dest_dir, 'X_train.npy'), X)
    np.save(os.path.join(dest_dir, 'y_train.npy'), y)
    print(f"\nProcessing complete! X={X.shape}, y={y.shape}")

if __name__ == '__main__':
    data_dir = 'dataset/processed_combine_asl_dataset'
    image_tuples = get_image_tuples(data_dir, max_per_class=200)
    print(f"Found {len(image_tuples)} images.")
    labels = sorted(list(set([label for _, label in image_tuples])))
    label_map = {lbl: idx for idx, lbl in enumerate(labels)}
    X, y = process_custom_dataset(image_tuples, label_map)
    save_custom_processed_data(X, y, label_map)
