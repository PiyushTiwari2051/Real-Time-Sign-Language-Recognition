# SignBridge Pro - Real-Time Sign Language Recognition System

SignBridge Pro is a real-time American Sign Language (ASL) word recognition system built on Google's Kaggle ASL Signs dataset and modern deep learning research. 

---

#🌟 Key Features

**Skeleton-Only Visualization**: The raw webcam feed is hidden. Instead, hand skeleton lines and pose joints are plotted on a black canvas. This serves as a research-grade visualization showing that inference runs purely on spatial landmark coordinates, rather than color details or background patterns.
**Temporal Attention Heatmap**: Shows a 30-bar real-time heatmap representing which video frames the model's self-attention mechanism evaluated as most significant for the classification. **Live Speedometer Confidence Gauge**: Renders a semicircular speedometer gauge showing prediction confidence.
 * *Below 60%*: Displays `...` (thinking).
 * *Above 60%*: Displays the predicted sign word.
 * *Above 85%*: Instantly triggers the text-to-speech speaker.
 * **Sentence Builder**: Recognized words with high confidence stack into a sentence. Pressing `SPACE` speaks the accumulated sentence, and pressing `C` clears it.
  * **Generalization Testing**: Prompt on startup logs predictions to a generalization text file (`results/generalisation_log.txt`) if a new, untrained signer is using the system, helping evaluate real-world accessibility robustness.

---

#🏗️ Folder Structure

```text
SignBridgePro/
│
├── data/
│   ├── raw/                     ← Downloaded Kaggle parquet files
│   │   └── train_landmark_files/
│   ├── processed/               ← Cleaned numpy sequences
│   │   ├── X_train.npy          ← Shape: (N, 30, 108)
│   │   ├── y_train.npy          ← Shape: (N,)
│   │   └── label_map.json       ← {0: "hello", 1: "thankyou", ...}
│   └── augmented/               ← Mirrored copies added here
│
├── src/
│   ├── step1_download_data.py   ← Kaggle API download + verify
│   ├── step2_process_kaggle.py  ← Parquet → sequences → numpy
│   ├── step3_augment.py         ← Mirror augmentation + save
│   ├── step4_train.py           ← Conv1D + BiLSTM + Attention train
│   ├── step5_evaluate.py        ← Confusion matrix + per-class acc
│   └── step6_live_demo.py       ← Webcam → MediaPipe → model → UI
│
├── model/
│   └── signbridge_best.pth      ← Saved best model weights
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curve.png
│   ├── per_class_accuracy.txt
│   └── generalisation_log.txt   ← Untrained signer prediction logs
│
└── utils/
    ├── keypoint_utils.py        ← Landmark extraction, normalization, and mirroring
    ├── display_utils.py         ← Custom skeleton and UI rendering components
    └── tts_utils.py             ← Asynchronous speech worker thread wrapper
```

---

#🔬 Model Architecture

```text
Input Sequence: (batch, 30 frames, 108 features)
      │
      ▼
[Transpose to (batch, 108, 30)]
      │
      ▼
[Conv1D Layer (64 filters, kernel=3, padding=same)] + ReLU
      │ (Extracts micro-temporal patterns across 3 consecutive frames)
      ▼
[Conv1D Layer (64 filters, kernel=3, padding=same)] + ReLU + Dropout(0.2)
      │
      ▼
[Transpose to (batch, 30, 64)]
      │
      ▼
[Bidirectional LSTM (128 units)]
      │ (Captures forward and backward global context memory)
      ▼
[BiLSTM output: (batch, 30, 256)]
      │
      ▼
[Self-Attention Layer (Linear layer → Softmax)]
      ├── Attention Weights: (batch, 30)  ──► Rendered as live heatmap
      ▼
[Context Vector (Attention-weighted sum): (batch, 256)]
      │
      ▼
[Dense Layer (128, ReLU)] + Dropout(0.3)
      │
      ▼
[Dense Output Layer (25)] ──► Logits for Cross-Entropy Loss
```

---

## 🚀 Setup & Execution Guide

### 1. Installation

Create a virtual environment and install the required packages:

```bash
# Activate virtual environment (Windows Powershell)
.\venv\Scripts\Activate.ps1

# Install package dependencies
pip install mediapipe opencv-python torch torchvision
pip install numpy pandas pyarrow matplotlib scikit-learn
pip install pyttsx3 tqdm kaggle
```

Configure your Kaggle credentials file:
1. Generate an API token from your Kaggle Account tab.
2. Place `kaggle.json` inside your user directory: `C:\Users\<UserName>\.kaggle\kaggle.json`.

### 2. Execution Pipeline

Run each pipeline step sequentially:

* **Step 1: Download & Verify Data**
  ```bash
  python src/step1_download_data.py
  ```
  Downloads the dataset competition files, extracts the dataset, and performs folder verification.

* **Step 2: Preprocess Landmark Sequences**
  ```bash
  python src/step2_process_kaggle.py
  ```
  Reads training metadata, filters files for 25 target signs, extracts right hand (or mirrored left hand) and upper body pose landmarks, and outputs them into numpy arrays.

* **Step 3: Apply Mirror Augmentation**
  ```bash
  python src/step3_augment.py
  ```
  Flips the processed coordinate datasets along the X-axis, doubling training data to support robust recognition for left and right hands.

* **Step 4: Train Deep Learning Model**
  ```bash
  python src/step4_train.py
  ```
  Trains the Conv1D + BiLSTM + Attention model on the augmented dataset and outputs training curves.

* **Step 5: Evaluate Performance**
  ```bash
  python src/step5_evaluate.py
  ```
  Computes validation accuracies, confusion matrix plots, and outputs detailed classification logs.

* **Step 6: Launch Live Camera Demo**
  ```bash
  python src/step6_live_demo.py
  ```
  Launches the live webcam interface utilizing MediaPipe and our trained classifier.
