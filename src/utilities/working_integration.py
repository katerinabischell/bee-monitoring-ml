#!/usr/bin/env python3
"""
Working integration script for your actual bee-monitoring-ml setup
Uses your correct_video_analyzer.py functions directly
"""

import os
import sys
import json
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Import your actual functions
try:
    from correct_video_analyzer import test_model_on_video, BombusClassifier
    print("✅ Successfully imported your video analysis functions")
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import video analyzer: {e}")
    ANALYZER_AVAILABLE = False

class WorkingBombusUpgrader:
    """
    Working upgrader using your actual video analysis functions
    """
    
    def __init__(self):
        self.model_path = "best_bombus_model.pth"
        self.video_base_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/birds_beak"
        
    def test_your_model(self, video_path):
        """
        Test your actual model on a video using your function
        """
        if not ANALYZER_AVAILABLE:
            print("❌ Video analyzer not available")
            return None
            
        print(f"🎬 Testing your model on: {os.path.basename(video_path)}")
        
        try:
            # Use your actual function
            results = test_model_on_video(video_path, self.model_path)
            print("✅ Your model test successful!")
            return results
            
        except Exception as e:
            print(f"❌ Error running your model: {e}")
            return None
    
    def simulate_5min_segments_from_your_results(self, video_path):
        """
        Use your actual model results to simulate 5-minute segment analysis
        """
        # Get results from your model
        results = self.test_your_model(video_path)
        if not results:
            return None
        
        # Your model returns detection info - let's extract it
        print("\n📊 ANALYZING YOUR MODEL OUTPUT:")
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        cap.release()
        
        print(f"   Video duration: {total_duration:.1f} seconds")
        print(f"   FPS: {fps:.1f}")
        print(f"   Total frames: {total_frames}")
        
        # Break into 5-minute segments
        segment_duration = 300  # 5 minutes
        num_segments = int(np.ceil(total_duration / segment_duration))
        
        segments = []
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            
            # For each segment, run your model on sample frames
            segment_detections = self.analyze_segment_with_your_model(
                video_path, start_time, end_time, fps
            )
            
            segments.append({
                'segment_number': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'detections': segment_detections,
                'bee_count': len([d for d in segment_detections if d['has_bombus']]),
                'avg_confidence': np.mean([d['confidence'] for d in segment_detections]) if segment_detections else 0,
                'detection_rate': len([d for d in segment_detections if d['has_bombus']]) / len(segment_detections) if segment_detections else 0
            })
            
            print(f"   Segment {i+1} ({start_time:.0f}-{end_time:.0f}s): {segments[-1]['bee_count']} detections")
        
        return segments
    
    def analyze_segment_with_your_model(self, video_path, start_time, end_time, fps):
        """
        Analyze a video segment using your trained model
        Sample frames every 10 seconds within the segment
        """
        if not ANALYZER_AVAILABLE:
            return []
        
        try:
            import torch
            import torchvision.transforms as transforms
            from PIL import Image
            
            # Load your model
            model = BombusClassifier(num_classes=2)
            model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            model.eval()
            
            # Image preprocessing (match your model's training)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            cap = cv2.VideoCapture(video_path)
            detections = []
            
            # Sample every 10 seconds within segment
            sample_interval = 10
            for t in range(int(start_time), int(end_time), sample_interval):
                frame_number = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert frame for model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                image_tensor = transform(pil_image).unsqueeze(0)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence = probabilities[0][1].item()  # Bombus probability
                    has_bombus = confidence > 0.5
                
                detections.append({
                    'timestamp': t,
                    'frame_number': frame_number,
                    'has_bombus': has_bombus,
                    'confidence': confidence
                })
            
            cap.release()
            return detections
            
        except Exception as e:
            print(f"⚠️ Error in segment analysis: {e}")
            return []
    
    def run_complete_analysis(self, video_path):
        """
        Run complete analysis: current model + 5-min segments simulation
        """
        print(f"\n{'='*60}")
        print(f"COMPLETE BOMBUS ANALYSIS")
        print(f"Video: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        # 1. Test your current model (like your training notes)
        print("\n1. Running your current model analysis...")
        current_results = self.test_your_model(video_path)
        
        # 2. Simulate 5-minute segment analysis
        print("\n2. Breaking into 5-minute segments...")
        segments = self.simulate_5min_segments_from_your_results(video_path)
        
        if segments:
            # 3. Generate summary
            print(f"\n3. Analysis Summary:")
            total_detections = sum(s['bee_count'] for s in segments)
            active_segments = len([s for s in segments if s['bee_count'] > 0])
            
            print(f"   📊 Total segments: {len(segments)}")
            print(f"   🐝 Segments with bees: {active_segments}")
            print(f"   🔢 Total bee detections: {total_detections}")
            print(f"   📈 Activity rate: {active_segments/len(segments)*100:.1f}%")
            
            # Find peak activity segment
            peak_segment = max(segments, key=lambda x: x['bee_count'])
            print(f"   🏆 Peak activity: Segment {peak_segment['segment_number']} ({peak_segment['bee_count']} detections)")
        
        # 4. Save results
        results = {
            'video_path': video_path,
            'analysis_date': datetime.now().isoformat(),
            'current_model_results': str(current_results) if current_results else None,
            'segments': segments,
            'summary': {
                'total_segments': len(segments) if segments else 0,
                'active_segments': len([s for s in segments if s['bee_count'] > 0]) if segments else 0,
                'total_detections': sum(s['bee_count'] for s in segments) if segments else 0
            }
        }
        
        # Save to file
        output_file = f"object_detection_results/{Path(video_path).stem}_analysis.json"
        os.makedirs("object_detection_results", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        return results


def test_with_your_video():
    """
    Test with your known good video from training notes
    """
    upgrader = WorkingBombusUpgrader()
    
    # Your test video from training notes (100% detection rate)
    video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/birds_beak/week_2/day_1/site_3/mid/P1000087.MP4"
    
    if os.path.exists(video_path):
        print("🎯 Testing with your known good video (P1000087.MP4)")
        results = upgrader.run_complete_analysis(video_path)
        return results
    else:
        print("❌ Video file not found. Check if external drive is mounted.")
        print("💡 Expected path:", video_path)
        return None

def batch_analyze_videos():
    """
    Analyze multiple videos from your collection
    """
    upgrader = WorkingBombusUpgrader()
    base_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/birds_beak"
    
    # Look for video files
    video_files = []
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.MP4') or file.endswith('.mp4'):
                    video_files.append(os.path.join(root, file))
    
    if not video_files:
        print("❌ No video files found. Check if external drive is mounted.")
        return
    
    print(f"📹 Found {len(video_files)} video files")
    
    # Analyze first 3 videos as test
    for i, video_path in enumerate(video_files[:3]):
        print(f"\n{'='*60}")
        print(f"Processing video {i+1}/3: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        results = upgrader.run_complete_analysis(video_path)

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_with_your_video()
        elif sys.argv[1] == "batch":
            batch_analyze_videos()
        elif sys.argv[1].endswith('.MP4') or sys.argv[1].endswith('.mp4'):
            # Analyze specific video
            upgrader = WorkingBombusUpgrader()
            upgrader.run_complete_analysis(sys.argv[1])
        else:
            print("Usage:")
            print("  python working_integration.py test           # Test with P1000087.MP4")
            print("  python working_integration.py batch          # Analyze multiple videos")  
            print("  python working_integration.py /path/to/video # Analyze specific video")
    else:
        # Default: test with your known good video
        test_with_your_video()

if __name__ == "__main__":
    main()