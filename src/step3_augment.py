# src/step3_augment.py
# This script applies mirror augmentation to the processed landmark sequences.
# It flips the coordinates horizontally, doubling the size of the training dataset.

import os
import numpy as np

# Import the mirroring utility function
from utils.keypoint_utils import mirror_sequence

def load_processed_data(src_dir):
    # Load original processed landmark sequences and label indices
    X = np.load(os.path.join(src_dir, 'X_train.npy'))
    y = np.load(os.path.join(src_dir, 'y_train.npy'))
    return X, y

def apply_mirror_augmentation(X):
    mirrored_list = []
    # Loop over all samples and apply left-right mirroring
    for i in range(len(X)):
        mirrored_seq = mirror_sequence(X[i])
        mirrored_list.append(mirrored_seq)
    # Convert list to numpy array
    return np.array(mirrored_list, dtype=np.float32)

def save_augmented_dataset(X_orig, X_mirr, y, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    # Concatenate original and mirrored sequences to double the dataset
    X_combined = np.concatenate([X_orig, X_mirr], axis=0)
    # Duplicate the labels accordingly
    y_combined = np.concatenate([y, y], axis=0)
    # Save the augmented datasets
    np.save(os.path.join(dest_dir, 'X_train.npy'), X_combined)
    np.save(os.path.join(dest_dir, 'y_train.npy'), y_combined)
    print(f"Augmented dataset saved. Shape: X={X_combined.shape}, y={y_combined.shape}")

if __name__ == '__main__':
    src = 'data/processed'
    dest = 'data/augmented'
    # Run augmentation pipeline
    if not os.path.exists(os.path.join(src, 'X_train.npy')):
        print("Processed data files not found. Please run step2_process_kaggle.py first.")
    else:
        print("Loading processed keypoint data...")
        X, y = load_processed_data(src)
        print(f"Applying mirror augmentation on {len(X)} samples...")
        X_mirrored = apply_mirror_augmentation(X)
        print("Combining and saving augmented dataset...")
        save_augmented_dataset(X, X_mirrored, y, dest)
