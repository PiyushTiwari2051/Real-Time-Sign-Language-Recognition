# src/setup_mock_demo.py
# This script generates synthetic landmark data to let you test the training,
# evaluation, and live webcam demo immediately without needing Kaggle API credentials.

import os
import json
import subprocess
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use the same target signs as our core pipeline
from src.step2_process_kaggle import TARGET_SIGNS

def generate_synthetic_data(num_samples_per_class=15):
    X, y = [], []
    # Loop over each of the 25 classes
    for class_id in range(25):
        for _ in range(num_samples_per_class):
            # Generate a baseline pattern with some random variations
            seq = np.random.normal(loc=class_id / 25.0, scale=0.05, size=(30, 108))
            # Keep coordinates in typical normalized range [0, 1]
            seq = np.clip(seq, 0.0, 1.0)
            X.append(seq)
            y.append(class_id)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def save_initial_files(X, y, dest_dir='data/processed'):
    os.makedirs(dest_dir, exist_ok=True)
    # Save processed dataset arrays
    np.save(os.path.join(dest_dir, 'X_train.npy'), X)
    np.save(os.path.join(dest_dir, 'y_train.npy'), y)
    # Save the label map JSON file
    label_map = {sign: idx for idx, sign in enumerate(TARGET_SIGNS)}
    with open(os.path.join(dest_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"Generated synthetic processed dataset: X shape={X.shape}, y shape={y.shape}")

def run_pipeline_steps():
    python_path = os.path.join('venv', 'Scripts', 'python.exe')
    # Set PYTHONPATH to root directory so python can find 'src' and 'utils' modules
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print("--- Running Step 3: Augment ---")
    subprocess.run([python_path, 'src/step3_augment.py'], env=env, check=True)
    print("--- Running Step 4: Train ---")
    subprocess.run([python_path, 'src/step4_train.py'], env=env, check=True)
    print("--- Running Step 5: Evaluate ---")
    subprocess.run([python_path, 'src/step5_evaluate.py'], env=env, check=True)

if __name__ == '__main__':
    print("Setting up synthetic workspace for immediate verification...")
    # Generate 15 samples per class
    X_synthetic, y_synthetic = generate_synthetic_data(15)
    save_initial_files(X_synthetic, y_synthetic)
    # Run augmentation, training, and evaluation scripts
    run_pipeline_steps()
    print("\nWorkspace initialized successfully!")
    print("To test the live webcam application, run:")
    print("python src/step6_live_demo.py")
