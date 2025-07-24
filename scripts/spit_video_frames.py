#install cv2 if not already installed
#python3 -m pip install opencv-python

#written by KC Seltmannn with ChatGPT near July 23, 2025
# spits video segments into single images and places them all into a single folder
# run in command line: python3 spit_video_frames.py
# resulting frames are 960 × 540 and ca. 188KB

import cv2
from pathlib import Path

input_dir = Path("/Volumes/IMAGES/bombus_synthetic_image_data/downsampled_video/SMBB")
output_dir = Path("/Volumes/IMAGES/bombus_synthetic_image_data/extracted_frames/SMBB")
output_dir.mkdir(exist_ok=True)

for video_path in input_dir.glob("*.MP4"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        continue

    frame_folder = output_dir / video_path.stem
    frame_folder.mkdir(parents=True, exist_ok=True)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = frame_folder / f"frame_{frame_num:05d}.jpg"
        cv2.imwrite(str(frame_file), frame)
        frame_num += 1

    cap.release()
    print(f"Extracted {frame_num} frames from {video_path.name}")
