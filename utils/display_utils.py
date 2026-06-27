# utils/display_utils.py
# This file provides functions to render the SignBridge Pro custom user interface.
# It handles drawing skeletons, confidence gauges, attention heatmaps, and sentence builders.

import cv2
import numpy as np

# Connections for drawing the hand skeleton (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky finger
    (0, 17)                              # Wrist connection
]

# Connections for upper body pose (15 landmarks)
POSE_CONNECTIONS = [
    (11, 12),                            # Shoulder to shoulder
    (11, 13), (12, 14),                  # Shoulders to elbows
    (0, 11), (0, 12)                     # Nose to shoulders
]

def draw_landmark_nodes(canvas, coords, start_idx, end_idx, color):
    # Iterate over the specified range of landmarks
    for idx in range(start_idx, end_idx):
        x, y = int(coords[idx][0] * 640), int(coords[idx][1] * 480)
        # Only draw if the landmark is not a padding placeholder
        if x > 0 and y > 0:
            cv2.circle(canvas, (x, y), 4, color, -1)

def draw_skeleton_lines(canvas, coords, connections, start_idx, color):
    # Draw connection lines between nodes
    for conn in connections:
        pt1 = coords[start_idx + conn[0]]
        pt2 = coords[start_idx + conn[1]]
        x1, y1 = int(pt1[0] * 640), int(pt1[1] * 480)
        x2, y2 = int(pt2[0] * 640), int(pt2[1] * 480)
        # Connect only if both landmarks are valid (non-zero)
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(canvas, (x1, y1), (x2, y2), color, 2)

def draw_skeleton(canvas, landmarks):
    # Reshape landmarks back to (36 landmarks, 3 coordinates)
    coords = landmarks.reshape((36, 3))
    # Draw hand nodes in lime green
    draw_landmark_nodes(canvas, coords, 0, 21, (0, 255, 128))
    # Draw hand lines in green
    draw_skeleton_lines(canvas, coords, HAND_CONNECTIONS, 0, (0, 200, 0))
    # Draw upper pose nodes in cyan
    draw_landmark_nodes(canvas, coords, 21, 36, (255, 255, 0))
    # Draw upper pose lines in blue-cyan
    draw_skeleton_lines(canvas, coords, POSE_CONNECTIONS, 21, (200, 200, 0))

def draw_confidence_gauge(canvas, confidence, center=(540, 100), radius=50):
    # Draw the gauge background arc (semi-circle)
    cv2.ellipse(canvas, center, (radius, radius), 0, 180, 360, (50, 50, 50), 6)
    # Highlight the active level on the arc in orange-red
    active_angle = int(confidence * 180)
    cv2.ellipse(canvas, center, (radius, radius), 0, 180, 180 + active_angle, (0, 165, 255), 6)
    # Calculate the angle of the needle
    needle_rad = np.pi - (confidence * np.pi)
    nx = int(center[0] + radius * np.cos(needle_rad))
    ny = int(center[1] - radius * np.sin(needle_rad))
    # Draw the speedometer needle
    cv2.line(canvas, center, (nx, ny), (255, 255, 255), 2)
    # Print the confidence text below the gauge
    cv2.putText(canvas, f"Conf: {int(confidence*100)}%", (center[0]-35, center[1]+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def draw_attention_heatmap(canvas, attention_weights, x_offset=20, y_offset=420, width=320, height=40):
    # Find the maximum weight to scale the visualization
    max_weight = np.max(attention_weights) + 1e-6
    bar_width = width // len(attention_weights)
    # Draw each frame's attention weight as a bar
    for idx, weight in enumerate(attention_weights):
        norm = weight / max_weight
        bar_height = int(norm * height)
        # Calculate color transition from blue (low) to red (high)
        color = (int((1-norm)*255), 0, int(norm*255))
        bx1 = x_offset + idx * bar_width
        by1 = y_offset - bar_height
        cv2.rectangle(canvas, (bx1, by1), (bx1 + bar_width - 1, y_offset), color, -1)
    # Draw heatmap label
    cv2.putText(canvas, "Temporal Attention Weights", (x_offset, y_offset + 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

def draw_sentence_builder(canvas, words, x_offset=20, y_offset=450, width=600):
    # Render translucent background panel
    panel = canvas[y_offset:y_offset+25, x_offset:x_offset+width]
    cv2.addWeighted(panel, 0.5, np.zeros_like(panel), 0.5, 0, panel)
    # Join list of words into a single sentence
    sentence = " ".join(words) if words else "[No signs recorded]"
    # Draw the constructed sentence text
    cv2.putText(canvas, f"Sentence: {sentence}", (x_offset+10, y_offset+17), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

def draw_citation_watermark(canvas):
    # Print research paper references in the bottom-right corner
    watermark_text = "Architecture: Conv1D + BiLSTM + Self-Attention | Data: Google ASL Kaggle"
    cv2.putText(canvas, watermark_text, (180, 475), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)
