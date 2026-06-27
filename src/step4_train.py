# src/step4_train.py
# This script implements and trains the Conv1D + BiLSTM + Self-Attention model.
# It splits the augmented dataset, runs the training loop, and saves the best model.

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class SignBridgeModel(nn.Module):
    def __init__(self, num_classes=25):
        super().__init__()
        # Conv1D layers to learn temporal micro-motion across frames
        self.conv1 = nn.Conv1d(108, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(0.2)
        # BiLSTM layer to capture forward and backward sequence memory
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        # Attention projection layer to evaluate frame importance
        self.attention_fc = nn.Linear(256, 1)
        # Final classification layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Permute input from (batch, 30, 108) to (batch, 108, 30) for Conv1D
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout_conv(self.relu(self.conv2(x)))
        # Permute back to (batch, 30, 64) for LSTM input
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        # Compute self-attention weights (batch, 30, 1)
        attn_scores = self.attention_fc(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        # Compute weighted sum across frames (batch, 256)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        # Dense classification layers
        out = self.dropout_fc(self.relu(self.fc1(context)))
        logits = self.fc2(out)
        # Return both logits for loss calculation and attention weights for visualization
        return logits, attn_weights.squeeze(-1)

def prepare_loaders(data_dir, batch_size=64):
    # Load augmented numpy dataset
    X = np.load(os.path.join(data_dir, 'X_train.npy'))
    y = np.load(os.path.join(data_dir, 'y_train.npy'))
    # Perform 80-20 train-validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Convert datasets to PyTorch tensors
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    # Process each batch
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        # Forward pass
        logits, _ = model(X_b)
        loss = criterion(logits, y_b)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # Compute metrics
        total_loss += loss.item() * len(y_b)
        correct += (logits.argmax(dim=1) == y_b).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    # Disable gradient tracking for validation
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits, _ = model(X_b)
            loss = criterion(logits, y_b)
            # Accumulate loss and accuracy
            total_loss += loss.item() * len(y_b)
            correct += (logits.argmax(dim=1) == y_b).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def plot_curves(history, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    # Plot loss and accuracy curves side-by-side
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy Curve'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curve.png'))
    plt.close()

def run_training(model, train_loader, val_loader, device, epochs=100, patience=15):
    # Setup loss, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    # Main training loop
    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        # Store history
        for k, v in zip(history.keys(), [tr_loss, val_loss, tr_acc, val_acc]):
            history[k].append(v)
        print(f"Epoch {epoch+1:02d}: Train Loss={tr_loss:.4f}, Train Acc={tr_acc:.2%} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}")
        # Save best model weight checkpoint
        if val_loss < best_val_loss:
            best_val_loss, early_stop_counter = val_loss, 0
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), 'model/signbridge_best.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break
    return history

if __name__ == '__main__':
    data_dir = 'data/augmented'
    if not os.path.exists(os.path.join(data_dir, 'X_train.npy')):
        print("Augmented data files not found. Please run step3_augment.py first.")
    else:
        # Detect compute device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {device}")
        tr_loader, val_loader = prepare_loaders(data_dir)
        model = SignBridgeModel().to(device)
        hist = run_training(model, tr_loader, val_loader, device)
        plot_curves(hist)
        print("Training completed! Best checkpoint saved to model/signbridge_best.pth")
