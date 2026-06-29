# src/step6_live_demo.py
# This script runs the live webcam sign language recognition application.
# It captures video, extracts keypoints, runs inference, and renders the custom black canvas UI.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import collections
import datetime
import cv2
import numpy as np
import torch

# Import utility modules
from src.step4_train import SignBridgeModel
from utils.display_utils import draw_skeleton, draw_confidence_gauge, draw_attention_heatmap, draw_sentence_builder, draw_citation_watermark
from utils.tts_utils import speak_text

def get_hand_landmarks(hand_landmarks, is_left):
    if not hand_landmarks:
        return np.zeros((21, 3), dtype=np.float32)
    coords = []
    for lm in hand_landmarks.landmark:
        # Mirror the X coordinate for left hand
        x = 1.0 - lm.x if is_left else lm.x
        coords.append([x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)

def get_pose_landmarks(pose_landmarks):
    if not pose_landmarks:
        return np.zeros((15, 3), dtype=np.float32)
    coords = []
    # Extract the first 15 landmarks (upper body)
    for idx in range(15):
        lm = pose_landmarks.landmark[idx]
        coords.append([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)

def extract_live_keypoints(results):
    # Use right hand if available, otherwise left hand (mirrored)
    if results.right_hand_landmarks:
        hand_xyz = get_hand_landmarks(results.right_hand_landmarks, False)
    elif results.left_hand_landmarks:
        hand_xyz = get_hand_landmarks(results.left_hand_landmarks, True)
    else:
        hand_xyz = np.zeros((21, 3), dtype=np.float32)
    pose_xyz = get_pose_landmarks(results.pose_landmarks)
    return np.concatenate([hand_xyz.flatten(), pose_xyz.flatten()])

def run_inference(model, buffer, device):
    # Convert buffer to tensor and add batch dimension
    inp = torch.tensor(list(buffer), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, attn = model(inp)
        probs = torch.softmax(logits, dim=1)[0]
    class_id = probs.argmax().item()
    return class_id, probs[class_id].item(), attn[0].cpu().numpy()

def log_generalization(word, confidence, filepath='results/generalisation_log.txt'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Log prediction details to file
    with open(filepath, 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Pred: {word:<12} | Conf: {confidence:.2%}\n")

def process_predictions(class_id, conf, votes, sentence, gen_mode, classes):
    votes.append(class_id)
    # Perform majority voting over last 7 frames
    voted_id = collections.Counter(votes).most_common(1)[0][0]
    word = classes[voted_id]
    # Display '...' if confidence is low
    disp_word = word if conf >= 0.60 else "..."
    # Speak and add to sentence if confidence is high
    if conf >= 0.85 and (not sentence or sentence[-1] != word):
        sentence.append(word)
        speak_text(word)
        if gen_mode:
            log_generalization(word, conf)
    return disp_word

def process_frame(frame, holistic, buffer, model, device, votes, sentence, gen_mode, classes):
    # Mirror the frame horizontally for natural mirror interaction
    frame = cv2.flip(frame, 1)
    # Convert mirrored frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)
    features = extract_live_keypoints(results)
    buffer.append(features)
    # Initialize variables for return
    disp_word, conf, attn = "...", 0.0, np.ones(30) / 30.0
    if len(buffer) == 30:
        class_id, conf, attn = run_inference(model, buffer, device)
        disp_word = process_predictions(class_id, conf, votes, sentence, gen_mode, classes)
    
    # Create canvas by overlaying a dark translucent layer on top of the webcam feed
    canvas = frame.copy()
    overlay = np.zeros_like(canvas)
    cv2.addWeighted(canvas, 0.45, overlay, 0.55, 0, canvas)
    
    # Draw sleek glass HUD header panel at the top
    from utils.display_utils import draw_glass_panel, draw_glowing_text
    draw_glass_panel(canvas, (15, 10), (625, 80), (12, 12, 12), (0, 180, 255), 0.6)
    draw_glowing_text(canvas, "SIGNBRIDGE PRO", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, (0, 120, 255))
    draw_glowing_text(canvas, f"Sign: {disp_word}", (280, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2, (0, 120, 0))
    
    draw_skeleton(canvas, features)
    draw_confidence_gauge(canvas, conf)
    draw_attention_heatmap(canvas, attn)
    draw_sentence_builder(canvas, sentence)
    draw_citation_watermark(canvas)
    return canvas

def main_loop(model, holistic, cap, device, gen_mode, classes):
    # sliding buffers
    buffer = collections.deque(maxlen=30)
    votes = collections.deque(maxlen=7)
    sentence = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process keypoints and render canvas
        canvas = process_frame(frame, holistic, buffer, model, device, votes, sentence, gen_mode, classes)
        cv2.imshow("SignBridge Pro", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            break
        elif key == 32:  # Space
            speak_text(" ".join(sentence))
        elif key == ord('c') or key == ord('C'):  # Clear
            sentence.clear()

if __name__ == '__main__':
    # Prompt user for generalization logging setup
    signer_resp = input("Are you the training signer? (Y/N): ").strip().lower()
    gen_mode = True if signer_resp == 'n' else False
    if gen_mode:
        print("Generalization mode ACTIVE. Logging predictions to results/generalisation_log.txt")
    from mediapipe.python.solutions import holistic as mp_holistic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load custom classes dynamically from label map
    import json
    with open(os.path.join('data/processed', 'label_map.json'), 'r') as f:
        label_map = json.load(f)
    classes = sorted(list(label_map.keys()), key=lambda x: label_map[x])
    num_classes = len(label_map)
    # Load model and weights based on class count
    model = SignBridgeModel(num_classes=num_classes).to(device)
    if os.path.exists('model/signbridge_best.pth'):
        model.load_state_dict(torch.load('model/signbridge_best.pth', map_location=device))
    model.eval()
    # Initialize MediaPipe Holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    main_loop(model, holistic, cap, device, gen_mode, classes)
    cap.release()
    cv2.destroyAllWindows()
