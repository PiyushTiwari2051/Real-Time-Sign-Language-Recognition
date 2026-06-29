# utils/display_utils.py
# This file provides functions to render the SignBridge Pro premium user interface.
# It handles drawing skeleton joints matching the custom dataset, glass panels, neon indicators, and autocorrect.

import cv2
import numpy as np
import time

# Colors matching the custom drawing dataset joints (BGR format)
WRIST_COLOR = (48, 48, 255)       # Red
THUMB_COLOR = (180, 229, 255)     # Light Yellow/Orange
INDEX_COLOR = (21, 101, 192)      # Brownish Orange
MIDDLE_COLOR = (48, 255, 48)      # Bright Green
RING_COLOR = (255, 204, 0)        # Cyan
PINKY_COLOR = (128, 64, 128)      # Purple

# Finger-specific connections
FINGER_CONNECTIONS = [
    # Thumb (Light Yellow)
    ((0, 1), WRIST_COLOR), ((1, 2), THUMB_COLOR), ((2, 3), THUMB_COLOR), ((3, 4), THUMB_COLOR),
    # Index (Brownish Orange)
    ((0, 5), WRIST_COLOR), ((5, 6), INDEX_COLOR), ((6, 7), INDEX_COLOR), ((7, 8), INDEX_COLOR),
    # Middle (Bright Green)
    ((5, 9), INDEX_COLOR), ((9, 10), MIDDLE_COLOR), ((10, 11), MIDDLE_COLOR), ((11, 12), MIDDLE_COLOR),
    # Ring (Cyan)
    ((9, 13), MIDDLE_COLOR), ((13, 14), RING_COLOR), ((14, 15), RING_COLOR), ((15, 16), RING_COLOR),
    # Pinky (Purple)
    ((13, 17), RING_COLOR), ((17, 18), PINKY_COLOR), ((18, 19), PINKY_COLOR), ((19, 20), PINKY_COLOR),
    # Wrist boundary
    ((0, 17), WRIST_COLOR)
]

COMMON_WORDS = [
    "piyush", "hello", "yes", "no", "please", "thank", "you", "thanks", "welcome",
    "good", "morning", "afternoon", "evening", "night", "goodbye", "bye", "help",
    "deaf", "hearing", "sign", "language", "friend", "family", "mother", "father",
    "brother", "sister", "home", "work", "school", "happy", "sad", "angry", "tired",
    "love", "like", "want", "need", "have", "understand", "know", "how", "what",
    "where", "when", "why", "who", "which", "time", "day", "week", "month", "year",
    "eat", "drink", "water", "food", "more", "stop", "go", "come", "see", "hear",
    "speak", "read", "write", "learn", "teach", "sorry", "excuse", "me",
    "fine", "bad", "great", "awesome", "beautiful", "nice", "meet", "introduce",
    "myself", "name", "is", "my", "your", "his", "her", "their", "our", "we", "they"
]

def get_word_suggestions(prefix, limit=3):
    if not prefix:
        return []
    prefix = prefix.strip().lower()
    matches = [w for w in COMMON_WORDS if w.startswith(prefix)]
    matches = sorted(matches, key=len)
    return matches[:limit]

def draw_glass_panel(canvas, pt1, pt2, bg_color=(10, 10, 10), border_color=(0, 255, 128), alpha=0.6, border_thickness=1):
    x1, y1 = pt1
    x2, y2 = pt2
    # Crop panel region
    sub = canvas[y1:y2, x1:x2]
    rect = np.full_like(sub, bg_color)
    cv2.addWeighted(sub, 1.0 - alpha, rect, alpha, 0, sub)
    # Draw panel borders with modern rounded rectangles
    cv2.rectangle(canvas, pt1, pt2, border_color, border_thickness, cv2.LINE_AA)

def draw_glowing_text(canvas, text, pos, font, scale, color, thickness=1, glow_color=(0, 100, 0)):
    # Render glowing outline first
    cv2.putText(canvas, text, pos, font, scale, glow_color, thickness + 2, cv2.LINE_AA)
    # Render crisp foreground text
    cv2.putText(canvas, text, pos, font, scale, color, thickness, cv2.LINE_AA)

def draw_skeleton(canvas, landmarks):
    coords = landmarks.reshape((36, 3))
    # Check if hand coordinates are active (non-zero)
    hand_active = np.any(coords[0:21] != 0.0)
    if not hand_active:
        return
        
    # Draw custom colored skeleton connections
    for conn, color in FINGER_CONNECTIONS:
        pt1 = coords[conn[0]]
        pt2 = coords[conn[1]]
        x1, y1 = int(pt1[0] * 640), int(pt1[1] * 480)
        x2, y2 = int(pt2[0] * 640), int(pt2[1] * 480)
        if 0 < x1 < 640 and 0 < y1 < 480 and 0 < x2 < 640 and 0 < y2 < 480:
            cv2.line(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            
    # Draw custom colored joint nodes
    for idx in range(21):
        x, y = int(coords[idx][0] * 640), int(coords[idx][1] * 480)
        if 0 < x < 640 and 0 < y < 480:
            # Assign color dynamically based on joint group
            if idx == 0:
                color = WRIST_COLOR
            elif 1 <= idx <= 4:
                color = THUMB_COLOR
            elif 5 <= idx <= 8:
                color = INDEX_COLOR
            elif 9 <= idx <= 12:
                color = MIDDLE_COLOR
            elif 13 <= idx <= 16:
                color = RING_COLOR
            else:
                color = PINKY_COLOR
            cv2.circle(canvas, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)

def draw_confidence_gauge(canvas, confidence, center=(575, 45), radius=28):
    # Draw arc background track
    cv2.ellipse(canvas, center, (radius, radius), 0, 180, 360, (40, 40, 40), 5, cv2.LINE_AA)
    # Draw active gauge colored arc (Electric Blue to Emerald Green)
    color = (0, 255, 128) if confidence >= 0.8 else (255, 120, 0)
    active_angle = int(confidence * 180)
    cv2.ellipse(canvas, center, (radius, radius), 0, 180, 180 + active_angle, color, 5, cv2.LINE_AA)
    # Draw needle indicator
    needle_rad = np.pi - (confidence * np.pi)
    nx = int(center[0] + radius * np.cos(needle_rad))
    ny = int(center[1] - radius * np.sin(needle_rad))
    cv2.line(canvas, center, (nx, ny), (255, 255, 255), 2, cv2.LINE_AA)
    # Print label text
    cv2.putText(canvas, f"Conf: {int(confidence*100)}%", (center[0]-30, center[1]+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

def draw_attention_heatmap(canvas, attention_weights, x_offset=20, y_offset=425, width=280, height=35):
    # Draw a smooth glowing oscilloscope line instead of flat blocks
    max_weight = np.max(attention_weights) + 1e-6
    points = []
    bar_width = width / len(attention_weights)
    for idx, weight in enumerate(attention_weights):
        norm = weight / max_weight
        x = int(x_offset + idx * bar_width)
        y = int(y_offset - norm * height)
        points.append((x, y))
        
    # Draw oscilloscope track
    for idx in range(len(points) - 1):
        # Neon purple to red gradient line
        cv2.line(canvas, points[idx], points[idx+1], (255, 0, 128), 2, cv2.LINE_AA)
        
    # Draw label
    cv2.putText(canvas, "Temporal Attention Waves", (x_offset, y_offset + 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

def draw_hud_keyboard_suggestions(canvas, current_word, suggestions, x_offset=20, y_offset=385, width=600):
    if not current_word:
        return
    # Draw autocomplete suggestions panel
    draw_glass_panel(canvas, (x_offset, y_offset), (x_offset+width, y_offset+20), (5, 5, 5), (255, 120, 0), 0.5, 1)
    
    text = "Autocomplete: "
    if suggestions:
        text += f"[Tab] {suggestions[0].upper()}"
        for i, sug in enumerate(suggestions[1:], 1):
            text += f"  |  [{i}] {sug.lower()}"
    else:
        text += "[No word match]"
        
    cv2.putText(canvas, text, (x_offset+10, y_offset+13), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 180), 1, cv2.LINE_AA)

def draw_sentence_builder(canvas, sentence_words, current_word, x_offset=20, y_offset=442, width=600):
    draw_glass_panel(canvas, (x_offset, y_offset), (x_offset+width, y_offset+30), (12, 12, 12), (0, 180, 255), 0.7, 1)
    
    # Blinking typing cursor animation
    cursor = "_" if int(time.time() * 2) % 2 == 0 else ""
    sentence = " ".join(sentence_words)
    if sentence and current_word:
        full_text = f"Sentence: {sentence} {current_word}{cursor}"
    elif current_word:
        full_text = f"Sentence: {current_word}{cursor}"
    elif sentence:
        full_text = f"Sentence: {sentence}{cursor}"
    else:
        full_text = f"Sentence: {cursor}"
        
    cv2.putText(canvas, full_text, (x_offset+12, y_offset+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

def draw_citation_watermark(canvas):
    watermark_text = "Conv1D + BiLSTM + Attention  |  Mirror Invariant  |  Press [Space] to space, [Enter] to Speak"
    cv2.putText(canvas, watermark_text, (110, 483), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (110, 110, 110), 1, cv2.LINE_AA)
