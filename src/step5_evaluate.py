# src/step5_evaluate.py
# This script evaluates the trained model on the validation set.
# It computes validation metrics, a confusion matrix plot, and per-class accuracy reports.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Import our custom model architecture
from src.step4_train import SignBridgeModel
from src.step2_process_kaggle import TARGET_SIGNS

def load_validation_loader(data_dir, batch_size=64):
    # Load augmented dataset
    X = np.load(os.path.join(data_dir, 'X_train.npy'))
    y = np.load(os.path.join(data_dir, 'y_train.npy'))
    # Use same seed (42) as training to get the exact same validation split
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False)

def evaluate_model(model_path, loader, device):
    # Initialize and load model weights
    model = SignBridgeModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            logits, _ = model(X_b)
            # Collect targets and model predictions
            y_true.extend(y_b.numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
    return np.array(y_true), np.array(y_pred)

def save_accuracies(y_true, y_pred, classes, dest_file):
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    with open(dest_file, 'w') as f:
        # Calculate accuracy for each target sign category
        for idx, sign in enumerate(classes):
            class_mask = (y_true == idx)
            # Avoid division by zero if class is missing
            class_acc = (y_pred[class_mask] == idx).mean() if class_mask.sum() > 0 else 0.0
            f.write(f"Sign: {sign:<12} | Accuracy: {class_acc:.2%}\n")
            print(f"Sign: {sign:<12} | Accuracy: {class_acc:.2%}")

def plot_confusion(y_true, y_pred, classes, dest_img):
    # Compute standard confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    # Render matrix representation
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Validation Set')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(dest_img)
    plt.close()

if __name__ == '__main__':
    data_path = 'data/augmented'
    chkpt_path = 'model/signbridge_best.pth'
    # Run evaluation pipeline
    if not os.path.exists(chkpt_path):
        print("Trained model weights not found. Please run step4_train.py first.")
    else:
        # Detect compute device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Loading validation dataset...")
        val_loader = load_validation_loader(data_path)
        print("Evaluating model...")
        y_t, y_p = evaluate_model(chkpt_path, val_loader, device)
        print("Generating confusion matrix plot...")
        plot_confusion(y_t, y_p, TARGET_SIGNS, 'results/confusion_matrix.png')
        print("Writing per-class accuracies to text log...")
        save_accuracies(y_t, y_p, TARGET_SIGNS, 'results/per_class_accuracy.txt')
        print("Evaluation complete! Results saved in results/")
