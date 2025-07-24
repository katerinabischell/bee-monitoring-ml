#install cv2 if not already installed
#python3 -m pip install opencv-python

#written by KC Seltmannn with ChatGPT near July 23, 2025
#script downsamples video to half the quality it was recorded
#takes a long time to downsample videos. Best to run overnight
# run in command line: python3 make-video-frames.py

import cv2
import os
from pathlib import Path

# Input and output folders
input_dir = Path("/Volumes/IMAGES/bombus_synthetic_image_data/video/SMBB")              # Folder with original .mp4 files
output_dir = Path("/Volumes/IMAGES/bombus_synthetic_image_data/downsampled_video/SMBB")        # Folder to save resized videos
output_dir.mkdir(exist_ok=True)

# Loop over all .mp4 files in the input folder
for video_path in input_dir.glob("*.MP4"):
    print(f"Downsampling: {video_path.name}")

    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        continue

    # Original frame dimensions
    orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    fourcc      = cv2.VideoWriter_fourcc(*'mp4v')  # Output codec

    # New frame size (50%)
    # change this to reduce more or less
    new_width = orig_width // 2
    new_height = orig_height // 2

    # Create VideoWriter to save the downsampled video
    output_path = output_dir / video_path.name
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_width, new_height))

    # Read and write frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        resized = cv2.resize(frame, (new_width, new_height))
        out.write(resized)

    # Release resources
    cap.release()
    out.release()
    print(f"Saved downsampled video: {output_path}")

print("All videos processed.")
