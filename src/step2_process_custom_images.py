# src/step2_process_custom_images.py
# This script extracts MediaPipe hand keypoints from static cropped image files
# and packages them into standard 30-frame sequence arrays for training.

import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from mediapipe.python.solutions import hands as mp_hands

def get_image_tuples(data_dir, max_per_class=200):
    image_tuples = []
    # Loop over class subdirectories (0-9, a-d)
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            count = 0
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Store path and label
                    image_tuples.append((os.path.join(class_dir, filename), class_name))
                    count += 1
                    if count >= max_per_class:
                        break
    return image_tuples

def extract_hand_coords(image_path, hands_solver):
    image = cv2.imread(image_path)
    if image is None:
        return None
    # Run MediaPipe Hands on static image
    results = hands_solver.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None
    # Take the first detected hand
    hand_lms = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand_lms.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)

def build_sequence(hand_coords):
    # Repeat the 63-feature hand coordinates 30 times
    sequence_hand = np.tile(hand_coords, (30, 1))
    # Pad with 45 zeros representing empty pose context (63 + 45 = 108 features)
    sequence_pose = np.zeros((30, 45), dtype=np.float32)
    return np.concatenate([sequence_hand, sequence_pose], axis=1)

def process_custom_dataset(image_tuples, label_map, hands_solver):
    X, y = [], []
    for img_path, label in tqdm(image_tuples, desc="Processing images"):
        hand_coords = extract_hand_coords(img_path, hands_solver)
        if hand_coords is not None:
            # Construct sequence and label
            seq = build_sequence(hand_coords)
            X.append(seq)
            y.append(label_map[label])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def save_custom_processed_data(X, y, label_map, dest_dir='data/processed'):
    os.makedirs(dest_dir, exist_ok=True)
    # Save the label map JSON file
    with open(os.path.join(dest_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
    # Save the keypoint and label arrays
    np.save(os.path.join(dest_dir, 'X_train.npy'), X)
    np.save(os.path.join(dest_dir, 'y_train.npy'), y)
    print(f"\nProcessing complete! X={X.shape}, y={y.shape}")

if __name__ == '__main__':
    data_dir = 'dataset/processed_combine_asl_dataset'
    image_tuples = get_image_tuples(data_dir)
    print(f"Found {len(image_tuples)} images.")
    # Unique labels
    labels = sorted(list(set([label for _, label in image_tuples])))
    label_map = {lbl: idx for idx, lbl in enumerate(labels)}
    # Initialize MediaPipe Hands in static image mode
    hands_solver = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    X, y = process_custom_dataset(image_tuples, label_map, hands_solver)
    save_custom_processed_data(X, y, label_map)
