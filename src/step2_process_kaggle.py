# src/step2_process_kaggle.py
# This script processes the raw Kaggle Parquet landmark files into numpy arrays.
# It reads train.csv, filters for 25 target signs, extracts keypoints, and saves them.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import our custom keypoint extraction utility
from utils.keypoint_utils import extract_sequence

TARGET_SIGNS = [
    "hello", "thankyou", "please", "sorry", "help",
    "yes", "no", "stop", "more", "water",
    "eat", "sleep", "go", "come", "home",
    "name", "friend", "love", "happy", "sad",
    "good", "bad", "where", "who", "how"
]

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    # Filter the metadata to only contain our target signs
    df_filtered = df[df['sign'].isin(TARGET_SIGNS)].copy()
    # Create a mapping from word labels to class index integers
    label_map = {sign: idx for idx, sign in enumerate(TARGET_SIGNS)}
    # Add a class column to the dataframe
    df_filtered['class_id'] = df_filtered['sign'].map(label_map)
    return df_filtered, label_map

def process_file(parquet_path):
    # Read the parquet file containing MediaPipe landmarks
    df_landmarks = pd.read_parquet(parquet_path)
    # Extract the sequence of 30 frames and 108 features
    return extract_sequence(df_landmarks)

def process_all_samples(df_meta, raw_dir, max_per_class=80):
    X, y = [], []
    # Process each sign category
    for sign in TARGET_SIGNS:
        # Get metadata rows for this sign and limit to max_per_class
        df_sign = df_meta[df_meta['sign'] == sign].head(max_per_class)
        print(f"Processing sign '{sign}' ({len(df_sign)} samples)...")
        for _, row in tqdm(df_sign.iterrows(), total=len(df_sign)):
            file_path = os.path.join(raw_dir, row['path'])
            try:
                features = process_file(file_path)
                X.append(features)
                y.append(row['class_id'])
            except Exception as e:
                # Print message and skip corrupt files
                print(f"Error processing {file_path}: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def save_data(X, y, label_map, dest_dir='data/processed'):
    os.makedirs(dest_dir, exist_ok=True)
    # Save the label map as a JSON file
    with open(os.path.join(dest_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
    # Save the keypoints array and target label indices as numpy files
    np.save(os.path.join(dest_dir, 'X_train.npy'), X)
    np.save(os.path.join(dest_dir, 'y_train.npy'), y)
    print(f"Processed datasets saved: X shape={X.shape}, y shape={y.shape}")

if __name__ == '__main__':
    raw_path = 'data/raw'
    csv_file = os.path.join(raw_path, 'train.csv')
    # Run dataset processing pipeline
    if not os.path.exists(csv_file):
        print("Raw train.csv not found. Please run step1_download_data.py first.")
    else:
        print("Loading metadata...")
        meta_df, l_map = load_metadata(csv_file)
        print("Extracting features from parquet files...")
        X_data, y_data = process_all_samples(meta_df, raw_path)
        save_data(X_data, y_data, l_map)
