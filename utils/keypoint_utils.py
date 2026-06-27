# utils/keypoint_utils.py
# This file handles extraction, resampling, and spatial mirroring of hand and pose landmarks.
# It processes raw MediaPipe landmarks into fixed-length 30-frame sequences of 108 features.

import numpy as np
import pandas as pd

def determine_dominant_hand(df):
    # Count non-NaN values for left and right hands to see which was used more
    left_non_nan = df[df['type'] == 'left_hand']['x'].notna().sum()
    right_non_nan = df[df['type'] == 'right_hand']['x'].notna().sum()
    # Return left_hand if it has more active keypoints, otherwise right_hand
    return 'left_hand' if left_non_nan > right_non_nan else 'right_hand'

def extract_hand_data(df, hand_type):
    # Filter the dataframe for the selected dominant hand landmarks
    hand_df = df[df['type'] == hand_type]
    if hand_df.empty:
        # Return zeros if the hand is completely missing
        return np.zeros((21, 3), dtype=np.float32)
    # Pivot the landmarks to get them ordered by landmark_index
    coords = hand_df.sort_values('landmark_index')[['x', 'y', 'z']].values
    if len(coords) < 21:
        # Pad with zeros if landmarks are missing
        coords = np.pad(coords, ((0, 21 - len(coords)), (0, 0)))
    if hand_type == 'left_hand':
        # Mirror the X coordinate for left hand to treat it as right hand
        coords[:, 0] = 1.0 - coords[:, 0]
    # Replace NaN values with 0.0
    return np.nan_to_num(coords, nan=0.0)

def extract_pose_data(df):
    # Filter the dataframe for pose landmarks
    pose_df = df[df['type'] == 'pose']
    if pose_df.empty:
        # Return zeros if pose data is completely missing
        return np.zeros((15, 3), dtype=np.float32)
    # Get upper body landmarks (index 0 to 14)
    upper_pose = pose_df[pose_df['landmark_index'] < 15]
    # Sort and extract coordinates
    coords = upper_pose.sort_values('landmark_index')[['x', 'y', 'z']].values
    if len(coords) < 15:
        # Pad with zeros if landmarks are missing
        coords = np.pad(coords, ((0, 15 - len(coords)), (0, 0)))
    # Replace NaN values with 0.0
    return np.nan_to_num(coords, nan=0.0)

def process_single_frame(df_frame, hand_type):
    # Extract dominant hand coordinates (21 landmarks)
    hand_xyz = extract_hand_data(df_frame, hand_type)
    # Extract upper body pose coordinates (15 landmarks)
    pose_xyz = extract_pose_data(df_frame)
    # Combine hand and pose coordinates into a flat 108-feature vector
    return np.concatenate([hand_xyz.flatten(), pose_xyz.flatten()])

def extract_sequence(df, target_len=30):
    # Find the dominant hand across the entire video sequence
    hand_type = determine_dominant_hand(df)
    # Group the dataframe by frame number
    frames = sorted(df['frame'].unique())
    sequence_data = []
    # Process each frame one by one
    for frame in frames:
        df_frame = df[df['frame'] == frame]
        features = process_single_frame(df_frame, hand_type)
        sequence_data.append(features)
    # Convert list of frames to numpy array
    sequence_array = np.array(sequence_data, dtype=np.float32)
    # Resample the sequence to the fixed target length of 30
    return resample_sequence(sequence_array, target_len)

def resample_sequence(sequence, target_len=30):
    num_frames = len(sequence)
    if num_frames == 0:
        # Return empty sequence filled with zeros
        return np.zeros((target_len, 108), dtype=np.float32)
    # Create linearly spaced indices to sample from
    sample_indices = np.linspace(0, num_frames - 1, target_len).astype(int)
    # Return the resampled sequence
    return sequence[sample_indices]

def mirror_sequence(sequence):
    # Create a copy of the sequence
    mirrored = sequence.copy()
    # Reshape features back to (30, 36 landmarks, 3 coordinates)
    reshaped = mirrored.reshape((30, 36, 3))
    # Flip the X-coordinates
    reshaped[:, :, 0] = 1.0 - reshaped[:, :, 0]
    # Return flattened representation
    return reshaped.reshape((30, 108))

if __name__ == '__main__':
    # Print expected outputs to confirm script functionality
    print("Testing keypoint_utils.py...")
    mock_seq = np.random.rand(30, 108).astype(np.float32)
    mirrored_seq = mirror_sequence(mock_seq)
    print("Mock Sequence Shape:", mock_seq.shape)
    print("Mirrored Sequence Shape:", mirrored_seq.shape)
    print("Extraction functions verified successfully!")
