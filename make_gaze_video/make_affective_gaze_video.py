import cv2
import argparse
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def read_emotibit(folder_location, slug):
    data_pts = ["EA"]  # We're focusing on GSR (EA) for this example
    dfs = {}

    for pt in data_pts:
        try:
            df = pd.read_csv(f"{folder_location}/{slug}_{pt}.csv")
            dfs[pt] = df
        except FileNotFoundError:
            print(f"Could not locate file {pt}")
    
    return dfs

def load_gaze_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.iloc[5:]
    df = df[df['confidence'] >= 0.95]

    gaze_x = []
    gaze_y = []

    for norm_pos in df['norm_pos']:
        x, y = ast.literal_eval(norm_pos)
        gaze_x.append(x)
        gaze_y.append(y)

    timestamps = df['timestamp'].values

    return gaze_x, gaze_y, timestamps

def overlay_circle(frame, center, radius=50, color=(0, 255, 0), alpha=0.7):
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def overlay_gsr_value(frame, gsr_value, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text = f"GSR: {gsr_value:.2f}"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10

    bg_rect = ((text_x - 5, text_y - text_size[1] - 5), 
               (text_x + text_size[0] + 5, text_y + 5))
    cv2.rectangle(frame, bg_rect[0], bg_rect[1], (0, 0, 0), -1)

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

def scale_gsr(gsr_values, min_size, max_size):
    # Scaling the size based on GSR values (higher = bigger circle)
    scaled_size = np.interp(gsr_values, (min(gsr_values), max(gsr_values)), (min_size, max_size))

    # Custom logic to make higher values more red and lower more blue
    scaled_color = []
    for value in gsr_values:
        # Normalize the value between 0 and 1
        norm_value = (value - min(gsr_values)) / (max(gsr_values) - min(gsr_values))
        print(norm_value)

        # Red to blue interpolation (0 = blue, 1 = red)
        red = int(255 * norm_value)
        blue = int(255 * (1 - norm_value))
        color = (blue, 0, red)  # BGR (OpenCV format), interpolating between blue and red
        scaled_color.append(color)

    return scaled_size, np.array(scaled_color)



def process_video_with_gaze_and_gsr(video_path, output_path, gaze_x, gaze_y, timestamps, gsr_values, gsr_timestamps, fps, max_seconds=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    gaze_start_time = timestamps[0]
    print(f"Gaze start time: {gaze_start_time}")

    frame_idx = 0
    gaze_idx = 0
    max_frame_idx = None

    if max_seconds is not None:
        max_frame_idx = int(max_seconds * video_fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    gsr_size, gsr_color = scale_gsr(gsr_values, min_size=25, max_size=60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frame_idx is not None and frame_idx > max_frame_idx):
            break

        frame_time = frame_idx / video_fps + gaze_start_time

        while gaze_idx < len(timestamps) - 1 and abs(timestamps[gaze_idx + 1] - frame_time) < abs(timestamps[gaze_idx] - frame_time):
            gaze_idx += 1

        gsr_idx = np.searchsorted(gsr_timestamps, frame_time)
        if gsr_idx >= len(gsr_values):
            gsr_idx = len(gsr_values) - 1

        x = int(gaze_x[gaze_idx] * width)
        y = int((1 - gaze_y[gaze_idx]) * height)

        circle_size = gsr_size[gsr_idx]
        circle_color = tuple(int(c * 255) for c in gsr_color[gsr_idx][:3])

        if 0 <= x <= width and 0 <= y <= height:
            frame = overlay_circle(frame, (x, y), radius=int(circle_size), color=circle_color, alpha=0.7)

        #overlay_gsr_value(frame, gsr_values[gsr_idx], color=(255, 255, 255))

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Processing complete. Video saved to {output_path}")

def main(gaze_csv_path, gsr_folder, gsr_slug, video_path, output_video_path, fps=30):
    gaze_x, gaze_y, gaze_timestamps = load_gaze_data(gaze_csv_path)

    gsr_data = read_emotibit(gsr_folder, gsr_slug)
    gsr_df = gsr_data["EA"]
    gsr_values = gsr_df['EA'].values
    gsr_timestamps = gsr_df['LocalTimestamp'].values

    process_video_with_gaze_and_gsr(video_path, output_video_path, gaze_x, gaze_y, gaze_timestamps, gsr_values, gsr_timestamps, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gaze and GSR data, overlaying them on video.")
    parser.add_argument('--gaze', type=str, required=True, help='Path to the gaze data CSV file.')
    parser.add_argument('--gsr_folder', type=str, required=True, help='Path to the folder containing GSR data.')
    parser.add_argument('--gsr_slug', type=str, required=True, help='Slug of the GSR data files (e.g., session name).')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, default='output_with_gaze_and_gsr.mp4', help='Path to the output video file.')
    parser.add_argument('--fps', type=int, default=30, help='FPS of the gaze data for proper synchronization.')

    args = parser.parse_args()

    main(args.gaze, args.gsr_folder, args.gsr_slug, args.video, args.output, fps=args.fps)
