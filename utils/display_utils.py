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

def draw_glass_panel(canvas, pt1, pt2, bg_color=(15, 15, 15), border_color=(0, 255, 128), alpha=0.5):
    x1, y1 = pt1
    x2, y2 = pt2
    # Apply glassmorphic translucency
    sub = canvas[y1:y2, x1:x2]
    rect = np.full_like(sub, bg_color)
    cv2.addWeighted(sub, 1.0 - alpha, rect, alpha, 0, sub)
    # Draw border
    cv2.rectangle(canvas, pt1, pt2, border_color, 1)

def draw_glowing_text(canvas, text, pos, font, scale, color, thickness=1, glow_color=(0, 100, 0)):
    # Draw thicker neon background glow
    cv2.putText(canvas, text, pos, font, scale, glow_color, thickness + 2, cv2.LINE_AA)
    # Draw bright foreground text
    cv2.putText(canvas, text, pos, font, scale, color, thickness, cv2.LINE_AA)

def draw_landmark_nodes(canvas, coords, start_idx, end_idx, color, radius=3):
    for idx in range(start_idx, end_idx):
        x, y = int(coords[idx][0] * 640), int(coords[idx][1] * 480)
        if 0 < x < 640 and 0 < y < 480:
            cv2.circle(canvas, (x, y), radius, color, -1)

def draw_skeleton_lines(canvas, coords, connections, start_idx, color, thickness=2):
    for conn in connections:
        pt1 = coords[start_idx + conn[0]]
        pt2 = coords[start_idx + conn[1]]
        x1, y1 = int(pt1[0] * 640), int(pt1[1] * 480)
        x2, y2 = int(pt2[0] * 640), int(pt2[1] * 480)
        if 0 < x1 < 640 and 0 < y1 < 480 and 0 < x2 < 640 and 0 < y2 < 480:
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

def draw_skeleton(canvas, landmarks):
    coords = landmarks.reshape((36, 3))
    # Hand lines: clean cyan/teal
    draw_skeleton_lines(canvas, coords, HAND_CONNECTIONS, 0, (255, 200, 0), 2)
    # Hand joints: bright lime green
    draw_landmark_nodes(canvas, coords, 0, 21, (0, 255, 128), 3)
    
    # Calculate natural neck/shoulder skeleton lines dynamically
    nose = coords[21 + 0]
    l_shoulder = coords[21 + 11]
    r_shoulder = coords[21 + 12]
    l_elbow = coords[21 + 13]
    r_elbow = coords[21 + 14]
    
    x_ns, y_ns = int(nose[0] * 640), int(nose[1] * 480)
    x_ls, y_ls = int(l_shoulder[0] * 640), int(l_shoulder[1] * 480)
    x_rs, y_rs = int(r_shoulder[0] * 640), int(r_shoulder[1] * 480)
    x_le, y_le = int(l_elbow[0] * 640), int(l_elbow[1] * 480)
    x_re, y_re = int(r_elbow[0] * 640), int(r_elbow[1] * 480)
    
    color_line = (0, 180, 255)  # Electric orange/gold BGR
    # Draw body pose lines with clean vertical neck line
    if 0 < x_ls < 640 and 0 < y_ls < 480 and 0 < x_rs < 640 and 0 < y_rs < 480:
        cv2.line(canvas, (x_ls, y_ls), (x_rs, y_rs), color_line, 2)
        x_neck, y_neck = (x_ls + x_rs) // 2, (y_ls + y_rs) // 2
        if 0 < x_ns < 640 and 0 < y_ns < 480:
            cv2.line(canvas, (x_ns, y_ns), (x_neck, y_neck), color_line, 2)
        if 0 < x_le < 640 and 0 < y_le < 480:
            cv2.line(canvas, (x_ls, y_ls), (x_le, y_le), color_line, 2)
        if 0 < x_re < 640 and 0 < y_re < 480:
            cv2.line(canvas, (x_rs, y_rs), (x_re, y_re), color_line, 2)
            
    # Upper pose face joints (0 to 10): tiny hot magenta dots
    draw_landmark_nodes(canvas, coords, 21, 32, (180, 0, 255), 2)
    # Upper pose body joints (11 to 14): neon blue dots
    draw_landmark_nodes(canvas, coords, 32, 36, (255, 128, 0), 3)

def draw_confidence_gauge(canvas, confidence, center=(570, 45), radius=30):
    # Draw arc background
    cv2.ellipse(canvas, center, (radius, radius), 0, 180, 360, (60, 60, 60), 4)
    # Highlight active levels in bright orange-red
    active_angle = int(confidence * 180)
    cv2.ellipse(canvas, center, (radius, radius), 0, 180, 180 + active_angle, (0, 140, 255), 4)
    # Calculate speedometer needle position
    needle_rad = np.pi - (confidence * np.pi)
    nx = int(center[0] + radius * np.cos(needle_rad))
    ny = int(center[1] - radius * np.sin(needle_rad))
    cv2.line(canvas, center, (nx, ny), (255, 255, 255), 2)
    # Display confidence text
    cv2.putText(canvas, f"Conf: {int(confidence*100)}%", (center[0]-27, center[1]+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

def draw_attention_heatmap(canvas, attention_weights, x_offset=20, y_offset=425, width=280, height=35):
    max_weight = np.max(attention_weights) + 1e-6
    bar_width = width // len(attention_weights)
    # Render individual attention value bars
    for idx, weight in enumerate(attention_weights):
        norm = weight / max_weight
        bar_height = int(norm * height)
        color = (int((1-norm)*200), 0, int(norm*200))
        bx1 = x_offset + idx * bar_width
        by1 = y_offset - bar_height
        cv2.rectangle(canvas, (bx1, by1), (bx1 + bar_width - 1, y_offset), color, -1)
    cv2.putText(canvas, "Temporal Attention Weights", (x_offset, y_offset + 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA)

def draw_sentence_builder(canvas, words, x_offset=20, y_offset=445, width=600):
    draw_glass_panel(canvas, (x_offset, y_offset), (x_offset+width, y_offset+25), (10, 10, 10), (0, 180, 255), 0.6)
    sentence = " ".join(words) if words else "[No signs recorded]"
    cv2.putText(canvas, f"Sentence: {sentence}", (x_offset+10, y_offset+17), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

def draw_citation_watermark(canvas):
    watermark_text = "Architecture: Conv1D + BiLSTM + Self-Attention | Data: Google ASL Kaggle"
    cv2.putText(canvas, watermark_text, (200, 475), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1, cv2.LINE_AA)
