#!/usr/bin/env python3
"""
Debug video timing to understand the mismatch between manual annotations and extracted frames
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_video_timing():
    """Analyze the actual video to understand timing"""
    
    video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milkvetch/week 5/day 1/site 1/afternoon/P1000446.MP4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Open video and get properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"🎬 VIDEO ANALYSIS")
    print(f"{'='*50}")
    print(f"Video: {Path(video_path).name}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Test specific timestamps from your manual annotations
    test_timestamps = [
        (2, "1st bombus on the left"),
        (10, "2nd bombus flies over to 1st bombus"),
        (57, "3rd bombus seen on top flowers"),
        (242, "4th bombus very visible bottom right"),
        (326, "6th and 7th bombus clearly visible"),
        (364, "8th bombus clearly visible on left")
    ]
    
    print(f"\n🕐 TIMESTAMP VERIFICATION")
    print(f"Testing key timestamps from your manual annotations:")
    
    extracted_frames = []
    
    for timestamp, description in test_timestamps:
        # Calculate frame number
        frame_number = int(timestamp * fps)
        
        # Jump to that frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            # Save frame for inspection
            output_filename = f"debug_frame_{timestamp:04d}s.jpg"
            cv2.imwrite(output_filename, frame)
            extracted_frames.append((timestamp, description, output_filename, frame))
            print(f"   ✅ {timestamp:3d}s: Extracted frame -> {output_filename}")
        else:
            print(f"   ❌ {timestamp:3d}s: Could not extract frame")
    
    cap.release()
    
    # Display frames for visual inspection
    if extracted_frames:
        display_debug_frames(extracted_frames)
    
    return extracted_frames

def display_debug_frames(extracted_frames):
    """Display extracted frames for visual inspection"""
    
    num_frames = len(extracted_frames)
    if num_frames == 0:
        return
    
    # Create subplot grid
    cols = 3
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Debug Frames from Your Manual Annotation Timestamps', fontsize=16)
    
    for i, (timestamp, description, filename, frame) in enumerate(extracted_frames):
        row = i // cols
        col = i % cols
        
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(frame_rgb)
        axes[row, col].set_title(f"{timestamp}s: {description[:30]}", fontsize=10)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_frames, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n👀 VISUAL INSPECTION:")
    print(f"Look at the frames above. Do you see bees in any of them?")
    print(f"Compare with your manual annotations:")
    for timestamp, description, filename, _ in extracted_frames:
        print(f"   {timestamp:3d}s: {description}")

def find_actual_bee_activity():
    """Sample frames throughout the video to find where bees actually are"""
    
    video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milkvetch/week 5/day 1/site 1/afternoon/P1000446.MP4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\n🔍 SAMPLING VIDEO TO FIND ACTUAL BEE ACTIVITY")
    print(f"{'='*60}")
    
    # Sample every 30 seconds throughout the video
    sample_interval = 30  # seconds
    sample_timestamps = list(range(0, int(duration), sample_interval))
    
    print(f"Sampling {len(sample_timestamps)} frames every {sample_interval}s")
    
    sampled_frames = []
    
    for timestamp in sample_timestamps:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            output_filename = f"sample_frame_{timestamp:04d}s.jpg"
            cv2.imwrite(output_filename, frame)
            sampled_frames.append((timestamp, output_filename, frame))
            print(f"   Sample {timestamp:3d}s: {output_filename}")
    
    cap.release()
    
    # Display sample frames
    if len(sampled_frames) > 0:
        print(f"\n📸 Review these sample frames to find where bees are actually visible:")
        
        # Show first 6 samples
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Video Samples - Look for Bee Activity', fontsize=16)
        
        for i in range(min(6, len(sampled_frames))):
            timestamp, filename, frame = sampled_frames[i]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            row = i // 3
            col = i % 3
            axes[row, col].imshow(frame_rgb)
            axes[row, col].set_title(f"{timestamp}s", fontsize=12)
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(6, 6):
            if i < 6:
                row = i // 3
                col = i % 3
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return sampled_frames

def main():
    """Main debugging function"""
    
    print("🔧 VIDEO TIMING DEBUGGING")
    print("="*60)
    print("This will help us understand why extracted frames don't show bees")
    
    # Step 1: Analyze video properties and test specific timestamps
    print("\n1️⃣ Testing your manual annotation timestamps...")
    extracted_frames = analyze_video_timing()
    
    # Step 2: Sample throughout video to find actual bee activity
    print("\n2️⃣ Sampling video to find actual bee locations...")
    sampled_frames = find_actual_bee_activity()
    
    print(f"\n💡 DEBUGGING COMPLETE")
    print(f"{'='*50}")
    print(f"1. Check the debug frames above - do any show bees?")
    print(f"2. If no bees in debug frames, your timestamps might be off")
    print(f"3. Review sample frames to see where bees actually appear")
    print(f"4. We may need to re-watch the video and create new timestamps")
    
    # Clean up debug files
    cleanup = input("\nClean up debug image files? (y/n): ")
    if cleanup.lower() == 'y':
        for file in Path('.').glob('debug_frame_*.jpg'):
            file.unlink()
        for file in Path('.').glob('sample_frame_*.jpg'):
            file.unlink()
        print("🗑️ Debug files cleaned up")

if __name__ == "__main__":
    main()