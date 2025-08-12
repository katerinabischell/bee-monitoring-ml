#!/usr/bin/env python3
"""
Simple verification using only videos we know exist
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SimpleVerifier:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.results_dir = self.data_dir / "results" / "simple_verification"
        
        # Create output directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 SIMPLE VERIFICATION")
        print(f"Output directory: {self.results_dir}")
    
    def find_trained_model(self):
        """Find the best trained model"""
        model_path_v2 = self.models_dir / 'ventura_bee_detection_v2' / 'weights' / 'best.pt'
        if model_path_v2.exists():
            return model_path_v2
        return None
    
    def find_existing_videos(self):
        """Find videos that actually exist and were analyzed"""
        # Load the successful analysis results
        csv_path = self.data_dir / "results" / "video_analysis_expanded" / "expanded_detection_results.csv"
        
        if not csv_path.exists():
            print("❌ No analysis results found")
            return []
        
        df = pd.read_csv(csv_path)
        
        # Check which videos actually exist
        base_paths = [
            Path("/Volumes/Expansion1/summer2025_ncos_kb_collections/ventura_milkvetch"),
            Path("/Volumes/Expansion1/summer2025_ncos_kb_collections/ventura_milk_vetch"),  # Alternative spelling
        ]
        
        existing_videos = []
        
        for _, row in df.iterrows():
            video_name = row['video_name']
            week = row['week'] 
            day = row['day']
            site = row['site']
            period = row['period']
            
            # Try different path combinations
            for base_path in base_paths:
                potential_paths = [
                    base_path / f"week {week}" / f"day {day}" / f"site {site}" / period / video_name,
                    base_path / f"week{week}" / f"day{day}" / f"site{site}" / period / video_name,
                    base_path / f"week {week}" / f"day {day}" / f"site {site}" / f"{period}" / video_name,
                ]
                
                for video_path in potential_paths:
                    if video_path.exists():
                        existing_videos.append({
                            'path': video_path,
                            'video_name': video_name,
                            'week': week,
                            'day': day,
                            'site': site,
                            'period': period,
                            'row_data': row
                        })
                        break
                else:
                    continue
                break
        
        return existing_videos
    
    def verify_with_existing_videos(self, max_samples=10):
        """Verify detections using videos we can actually find"""
        print(f"\n🔍 FINDING EXISTING VIDEOS")
        
        existing_videos = self.find_existing_videos()
        
        if not existing_videos:
            print("❌ No existing videos found")
            print("Let's try a different approach...")
            
            # Alternative: Check what videos exist in your known working directory
            known_path = Path("/Volumes/Expansion1/summer2025_ncos_kb_collections/ventura_milkvetch/week 6/day 1/site 1")
            if known_path.exists():
                print(f"✅ Found known path: {known_path}")
                for period in ['morning', 'mid', 'afternoon']:
                    period_path = known_path / period
                    if period_path.exists():
                        videos = list(period_path.glob('*.MP4'))
                        for video in videos:
                            if not video.name.startswith('._'):
                                print(f"  Found: {video}")
                                # We can manually verify these
                                return self.manual_verify_known_video(video)
            return None
        
        print(f"✅ Found {len(existing_videos)} accessible videos")
        
        # Take a sample
        sample_videos = existing_videos[:max_samples]
        
        # Load model
        model_path = self.find_trained_model()
        if not model_path:
            return None
        
        model = YOLO(model_path)
        model.conf = 0.3
        
        print(f"\n🖼️ Verifying {len(sample_videos)} videos...")
        
        verification_results = []
        
        for i, video_info in enumerate(sample_videos):
            print(f"Processing {i+1}/{len(sample_videos)}: {video_info['video_name']}")
            
            result = self.verify_single_video(video_info, model)
            if result:
                verification_results.append(result)
        
        return verification_results
    
    def manual_verify_known_video(self, video_path):
        """Manually verify a video we know exists"""
        print(f"\n📹 MANUAL VERIFICATION: {video_path.name}")
        
        # Load model
        model_path = self.find_trained_model()
        if not model_path:
            return None
        
        model = YOLO(model_path)
        model.conf = 0.4
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample a few frames
        sample_frames = [0, total_frames//4, total_frames//2, 3*total_frames//4]
        
        for i, frame_num in enumerate(sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                timestamp = frame_num / fps
                
                # Run detection
                results = model(frame, verbose=False)
                boxes = results[0].boxes
                
                # Create visualization
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(frame_rgb)
                
                detections = 0
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                               linewidth=3, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
                        
                        ax.text(x1, y1-10, f'Bee: {conf:.3f}', color='red', fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
                        detections += 1
                
                ax.set_title(f"{video_path.name} - Frame {frame_num} - Time {timestamp:.1f}s - {detections} bees detected")
                ax.axis('off')
                
                # Save
                output_path = self.results_dir / f"manual_verify_{video_path.stem}_frame{frame_num}_bees{detections}.jpg"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  ✅ Frame {frame_num}: {detections} bees detected -> {output_path.name}")
        
        cap.release()
        print(f"✅ Manual verification complete - check {self.results_dir}")
        return True


def main():
    """Run simple verification"""
    verifier = SimpleVerifier()
    
    print("🔍 Attempting to verify detections with accessible videos...")
    
    results = verifier.verify_with_existing_videos(max_samples=5)
    
    if results:
        print(f"\n✅ Verification complete! Check {verifier.results_dir}")
    else:
        print("\n💡 Could not find accessible videos from the analysis.")
        print("The video analysis might have processed videos that are no longer accessible.")
        print("Check the file paths and video organization on your external drive.")

if __name__ == "__main__":
    main()