#!/usr/bin/env python3
"""
Check what frames were extracted and preview them
"""

import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def check_extracted_frames():
    """Check what frames were extracted from the dataset"""
    
    image_dir = Path("ventura_object_detection_dataset/images/train")
    
    if not image_dir.exists():
        print("❌ Dataset directory not found")
        return
    
    # Get all extracted frames
    image_files = sorted(list(image_dir.glob("*.jpg")))
    
    print(f"📊 EXTRACTED FRAMES SUMMARY")
    print(f"{'='*50}")
    print(f"Total frames extracted: {len(image_files)}")
    
    # Parse timestamps and check against manual annotations
    bee_timestamps = [2, 3, 4, 7, 10, 15, 16, 17, 27, 36, 38, 57, 62, 68, 69, 71, 
                     240, 242, 292, 315, 323, 325, 326, 336, 338, 355, 357, 364, 
                     369, 380, 386, 420, 422, 430, 433, 456, 471]
    
    extracted_timestamps = []
    bee_frames_found = []
    
    for img_file in image_files:
        try:
            # Extract timestamp from filename
            timestamp_str = img_file.stem.split('_t')[1]
            timestamp = int(timestamp_str.lstrip('0') or '0')
            extracted_timestamps.append(timestamp)
            
            if timestamp in bee_timestamps:
                bee_frames_found.append((timestamp, img_file.name))
        except:
            print(f"⚠️ Could not parse timestamp from: {img_file.name}")
    
    print(f"\n🐝 BEE FRAMES FOUND:")
    print(f"Expected bee frames: {len(bee_timestamps)}")
    print(f"Actual bee frames found: {len(bee_frames_found)}")
    
    if len(bee_frames_found) > 0:
        print(f"\nFound bee frames:")
        for timestamp, filename in bee_frames_found[:10]:  # Show first 10
            print(f"   {timestamp:3d}s: {filename}")
        if len(bee_frames_found) > 10:
            print(f"   ... and {len(bee_frames_found) - 10} more")
    
    # Check for key frames from manual annotations
    key_frames = {
        2: "1st bombus on the left",
        326: "6th and 7th bombus clearly visible (2 bees)",
        364: "8th bombus clearly visible on left",
        242: "4th bombus very visible bottom right"
    }
    
    print(f"\n🎯 KEY FRAME CHECK:")
    for timestamp, description in key_frames.items():
        filename = f"P1000446_ventura_t{timestamp:04d}.jpg"
        if any(str(timestamp) in str(t) for t in extracted_timestamps):
            print(f"   ✅ {timestamp:3d}s: {description}")
        else:
            print(f"   ❌ {timestamp:3d}s: {description} - NOT FOUND")
    
    return image_files, bee_frames_found

def preview_frames(image_files, num_preview=6):
    """Preview some extracted frames"""
    
    if len(image_files) == 0:
        print("No frames to preview")
        return
    
    print(f"\n🖼️ PREVIEWING FIRST {num_preview} FRAMES:")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Extracted Frame Preview', fontsize=16)
    
    for i in range(min(num_preview, len(image_files))):
        img_path = image_files[i]
        
        # Load and display image
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        row = i // 3
        col = i % 3
        axes[row, col].imshow(image_rgb)
        axes[row, col].set_title(img_path.name, fontsize=10)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_preview, 6):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Look for bees in these frames. Are they visible?")

def main():
    """Check extracted frames and preview them"""
    
    print("🔍 CHECKING EXTRACTED FRAMES")
    print("="*50)
    
    image_files, bee_frames_found = check_extracted_frames()
    
    if len(bee_frames_found) == 0:
        print("\n❌ No bee frames found! This suggests an issue with frame extraction.")
        print("💡 Possible causes:")
        print("   1. Wrong video was processed")
        print("   2. Timestamp extraction failed")
        print("   3. Video frame rate different than expected")
        return
    
    # Preview some frames
    preview_frames(image_files)
    
    print(f"\n📋 NEXT STEPS:")
    if len(bee_frames_found) > 0:
        print("1. If you can see bees in the preview, run the annotation tool:")
        print("   python3 simple_annotation_script.py")
        print("2. If no bees are visible, we may need to check video timestamps")
    else:
        print("1. Check if the correct video was processed")
        print("2. Verify manual annotations match the video")

if __name__ == "__main__":
    main()