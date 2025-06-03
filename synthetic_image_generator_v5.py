#Attempt to adjust foreground lighting to match background (very subtle)

import cv2
import os
import random
import numpy as np

# File paths
foreground_dir = r"C:\Users\ekros\OneDrive\Documents\Textbooks\Spring 2025\CCBER Machine Learning\Bee Stills"
background_dir = r"C:\Users\ekros\OneDrive\Documents\Textbooks\Spring 2025\CCBER Machine Learning\sample_images_unoccupied"
output_dir = r"C:\Users\ekros\OneDrive\Documents\Textbooks\Spring 2025\CCBER Machine Learning\synthetic_images"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Repeat the process 10 times
for i in range(1, 51):
    # Select a random PNG file from the foreground directory
    foreground_files = [f for f in os.listdir(foreground_dir) if f.lower().endswith('.png')]
    if not foreground_files:
        raise FileNotFoundError("No PNG files found in the foreground directory.")
    foreground_path = os.path.join(foreground_dir, random.choice(foreground_files))

    # Select a random BMP file from the background directory
    background_files = [f for f in os.listdir(background_dir) if f.lower().endswith('.bmp')]
    if not background_files:
        raise FileNotFoundError("No BMP files found in the background directory.")
    background_path = os.path.join(background_dir, random.choice(background_files))

    # Load images
    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
    background = cv2.imread(background_path)

    # Ensure foreground has an alpha channel
    if foreground.shape[2] != 4:
        raise ValueError("Foreground image must have an alpha channel.")

    # Generate a random scale factor between 0.1 and 1.0
    scale_factor = random.uniform(0.1, 1.0)

    # Calculate new dimensions for the foreground
    new_width = int(background.shape[1] * scale_factor)
    new_height = int(background.shape[0] * scale_factor)

    # Resize the foreground to the new dimensions
    foreground = cv2.resize(foreground, (new_width, new_height))

    # Randomly rotate the foreground
    angle = random.uniform(0, 360)
    rotation_matrix = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), angle, 1)
    foreground = cv2.warpAffine(foreground, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Adjust foreground lighting to match the background
    fg_bgr = foreground[:, :, :3]
    bg_bgr = background

    # Calculate mean and standard deviation for both images
    fg_mean, fg_std = cv2.meanStdDev(fg_bgr)
    bg_mean, bg_std = cv2.meanStdDev(bg_bgr)

    # Reshape mean and std to match the dimensions of the foreground
    fg_mean = fg_mean.reshape(1, 1, 3)
    fg_std = fg_std.reshape(1, 1, 3)
    bg_mean = bg_mean.reshape(1, 1, 3)
    bg_std = bg_std.reshape(1, 1, 3)

    # Safeguard: Clamp the standard deviation to a minimum value to avoid extreme scaling
    min_std = 10  # Minimum standard deviation to prevent over-amplification
    fg_std = np.maximum(fg_std, min_std)
    bg_std = np.maximum(bg_std, min_std)

    # Adjust the foreground mean to move closer to the background mean
    adjustment_strength = 0.3  # Controls how much the foreground is adjusted (lower = less adjustment)
    adjusted_fg_bgr = fg_bgr + adjustment_strength * (bg_mean - fg_mean)

    # Clip the adjusted foreground to valid pixel range
    adjusted_fg_bgr = np.clip(adjusted_fg_bgr, 0, 255).astype(np.uint8)

    # Blend the adjusted foreground with the original to reduce intensity
    blend_ratio = 0.5  # Adjust this value to control the intensity of the adjustment
    fg_bgr = cv2.addWeighted(fg_bgr, 1 - blend_ratio, adjusted_fg_bgr, blend_ratio, 0)

    # Split the foreground into its color and alpha channels
    alpha_channel = foreground[:, :, 3] / 255.0  # Normalize alpha to range [0, 1]

    # Randomly position the foreground on the background
    x_offset = random.randint(0, max(0, background.shape[1] - new_width))
    y_offset = random.randint(0, max(0, background.shape[0] - new_height))

    # Overlay the foreground on the background
    blended = background.copy()
    for c in range(0, 3):  # Iterate over color channels
        blended[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c] = (
            alpha_channel * fg_bgr[:, :, c] +
            (1 - alpha_channel) * blended[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c]
        )

    # Save the result with a unique name
    output_path = os.path.join(output_dir, f"synthetic_image5.{i}.png")
    cv2.imwrite(output_path, blended)

    print(f"Synthetic image saved to {output_path}")