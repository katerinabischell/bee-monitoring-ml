#!/usr/bin/env python3
"""
Fixed segment analyzer that properly analyzes each 5-minute segment independently
"""

import os
import sys
import json
import pandas as pd
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from datetime import datetime

# Import your model
try:
    from correct_video_analyzer import BombusClassifier
    print("✅ Successfully imported BombusClassifier")
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import BombusClassifier: {e}")
    MODEL_AVAILABLE = False

class ProperSegmentAnalyzer:
    """
    Properly analyze videos in 5-minute segments (not just repeat whole video analysis)
    """
    
    def __init__(self, model_path="best_bombus_model.pth"):
        self.model_path = model_path
        self.model = None
        self.transform = None
        self.load_model()
    
    def load_model(self):
        """Load your trained model once"""
        if not MODEL_AVAILABLE:
            print("❌ Model not available")
            return
        
        try:
            self.model = BombusClassifier(num_classes=2)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            
            # Preprocessing pipeline to match your training
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ Model loaded successfully for segment analysis")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def analyze_single_frame(self, frame):
        """
        Analyze a single frame for bombus presence
        Returns: (has_bombus: bool, confidence: float)
        """
        if self.model is None:
            return False, 0.0
        
        try:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply preprocessing
            image_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                bombus_prob = probabilities[0][1].item()
                has_bombus = bombus_prob > 0.5
            
            return has_bombus, bombus_prob
            
        except Exception as e:
            print(f"⚠️ Error analyzing frame: {e}")
            return False, 0.0
    
    def analyze_video_segment(self, video_path, start_time, end_time, sample_interval=10):
        """
        Analyze ONLY the specified time segment of the video
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            sample_interval: Sample frames every N seconds within segment
        
        Returns:
            List of detections within this segment only
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        detections = []
        
        # Sample frames only within this segment
        current_time = start_time
        while current_time < end_time:
            # Jump to specific time
            frame_number = int(current_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze this frame
            has_bombus, confidence = self.analyze_single_frame(frame)
            
            detections.append({
                'timestamp': current_time,
                'frame_number': frame_number,
                'has_bombus': has_bombus,
                'confidence': confidence
            })
            
            current_time += sample_interval
        
        cap.release()
        return detections
    
    def analyze_video_in_segments(self, video_path, segment_duration=300, sample_interval=10):
        """
        Break video into segments and analyze each independently
        
        Args:
            video_path: Path to video
            segment_duration: Length of each segment in seconds (default 5 minutes)
            sample_interval: Sample every N seconds within each segment
        """
        print(f"\n{'='*60}")
        print(f"PROPER SEGMENT ANALYSIS")
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Segment duration: {segment_duration}s ({segment_duration/60:.1f} minutes)")
        print(f"Sample interval: {sample_interval}s")
        print(f"{'='*60}")
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        cap.release()
        
        print(f"📊 Video info: {total_duration:.1f}s, {fps:.1f} FPS, {total_frames} frames")
        
        # Calculate number of segments
        num_segments = int(np.ceil(total_duration / segment_duration))
        print(f"🔪 Breaking into {num_segments} segments")
        
        all_segments = []
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            
            print(f"\n🔍 Analyzing Segment {i+1}: {start_time:.0f}s - {end_time:.0f}s")
            
            # Analyze only this segment
            segment_detections = self.analyze_video_segment(
                video_path, start_time, end_time, sample_interval
            )
            
            # Calculate segment statistics
            frames_with_bees = [d for d in segment_detections if d['has_bombus']]
            bee_count = len(frames_with_bees)
            total_frames_checked = len(segment_detections)
            avg_confidence = np.mean([d['confidence'] for d in frames_with_bees]) if frames_with_bees else 0
            detection_rate = bee_count / total_frames_checked if total_frames_checked > 0 else 0
            
            segment_summary = {
                'segment_number': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'frames_analyzed': total_frames_checked,
                'bee_detections': bee_count,
                'detection_rate': detection_rate,
                'avg_confidence': avg_confidence,
                'detections': segment_detections
            }
            
            all_segments.append(segment_summary)
            
            # Print segment results
            print(f"   📈 {bee_count}/{total_frames_checked} frames with bees ({detection_rate*100:.1f}%)")
            if bee_count > 0:
                print(f"   🎯 Average confidence: {avg_confidence:.3f}")
                # Show timing of detections in this segment
                detection_times = [d['timestamp'] for d in frames_with_bees]
                print(f"   ⏰ Bee activity at: {[f'{t:.0f}s' for t in detection_times[:5]]}")
                if len(detection_times) > 5:
                    print(f"        ... and {len(detection_times)-5} more")
            else:
                print(f"   ❌ No bee activity detected in this segment")
        
        # Overall summary
        print(f"\n{'='*60}")
        print(f"📊 OVERALL ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        total_bee_detections = sum(s['bee_detections'] for s in all_segments)
        segments_with_activity = len([s for s in all_segments if s['bee_detections'] > 0])
        overall_activity_rate = segments_with_activity / len(all_segments)
        
        print(f"Total segments: {len(all_segments)}")
        print(f"Segments with bee activity: {segments_with_activity}")
        print(f"Total bee detections: {total_bee_detections}")
        print(f"Overall activity rate: {overall_activity_rate*100:.1f}%")
        
        # Find most active segment
        most_active = max(all_segments, key=lambda x: x['bee_detections'])
        print(f"Most active segment: #{most_active['segment_number']} ({most_active['bee_detections']} detections)")
        
        # Save detailed results
        results = {
            'video_path': video_path,
            'analysis_date': datetime.now().isoformat(),
            'video_info': {
                'duration_seconds': total_duration,
                'fps': fps,
                'total_frames': total_frames
            },
            'analysis_parameters': {
                'segment_duration': segment_duration,
                'sample_interval': sample_interval,
                'num_segments': num_segments
            },
            'segments': all_segments,
            'summary': {
                'total_segments': len(all_segments),
                'segments_with_activity': segments_with_activity,
                'total_bee_detections': total_bee_detections,
                'overall_activity_rate': overall_activity_rate,
                'most_active_segment': most_active['segment_number']
            }
        }
        
        # Save to file
        output_file = f"object_detection_results/{Path(video_path).stem}_proper_segments.json"
        os.makedirs("object_detection_results", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return results
    
    def compare_with_original_analysis(self, video_path):
        """
        Compare segment analysis with your original whole-video approach
        """
        print(f"\n{'='*60}")
        print(f"COMPARISON: SEGMENTS vs ORIGINAL")
        print(f"{'='*60}")
        
        # Run segment analysis
        segment_results = self.analyze_video_in_segments(video_path)
        
        # Run your original analysis for comparison
        print(f"\n🔄 Running original analysis for comparison...")
        try:
            from correct_video_analyzer import test_model_on_video
            original_results = test_model_on_video(video_path)
            print("✅ Original analysis completed")
        except Exception as e:
            print(f"❌ Error running original analysis: {e}")
            original_results = None
        
        # Compare results
        print(f"\n📊 COMPARISON RESULTS:")
        if original_results:
            print(f"   Original method: 30 uniform samples across whole video")
        print(f"   Segment method: {segment_results['summary']['total_bee_detections']} detections across {segment_results['summary']['total_segments']} segments")
        print(f"   Segment activity rate: {segment_results['summary']['overall_activity_rate']*100:.1f}%")
        
        return segment_results, original_results


def main():
    """Main function to run proper segment analysis"""
    analyzer = ProperSegmentAnalyzer()
    
    # Test video path
    video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/birds_beak/week_2/day_1/site_3/mid/P1000087.MP4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print("💡 Make sure external drive is mounted")
        return
    
    # Run proper segment analysis
    results = analyzer.analyze_video_in_segments(
        video_path, 
        segment_duration=300,  # 5 minutes
        sample_interval=10     # Sample every 10 seconds
    )
    
    # Also run comparison
    analyzer.compare_with_original_analysis(video_path)
    
    print(f"\n✅ Analysis complete!")
    print(f"This shows ACTUAL bee activity patterns across 5-minute segments")
    print(f"Unlike your original method which samples the whole video uniformly")

if __name__ == "__main__":
    main()