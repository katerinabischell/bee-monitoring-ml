"""
Object Detection Analyzer
"""
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import detection
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class BombusObjectDetector:
    """
    Object detection model for counting and locating multiple bombus bees in video frames
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load pre-trained Faster R-CNN and modify for bombus detection
        self.model = detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Modify the classifier head for bombus detection (background + bombus)
        num_classes = 2  # background + bombus
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded trained model from {model_path}")
        else:
            print("⚠️ No trained model found. Initialize training first.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def detect_bombus_in_frame(self, frame: np.ndarray) -> Dict:
        """
        Detect bombus bees in a single frame
        Returns: Dictionary with detections, boxes, scores, and count
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform for model input
        image_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Extract predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold and bombus class (label = 1)
        valid_detections = (scores >= self.confidence_threshold) & (labels == 1)
        
        return {
            'boxes': boxes[valid_detections],
            'scores': scores[valid_detections],
            'count': np.sum(valid_detections),
            'frame_shape': frame.shape
        }
    
    def analyze_video_segments(self, video_path: str, segment_duration: int = 300) -> pd.DataFrame:
        """
        Analyze video in segments (default 5 minutes = 300 seconds)
        Returns DataFrame with bee counts per segment
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        
        print(f"🎬 Analyzing video: {os.path.basename(video_path)}")
        print(f"📊 Total duration: {total_duration:.1f}s, FPS: {fps:.1f}")
        print(f"🔪 Breaking into {segment_duration}s segments")
        
        results = []
        segment_number = 0
        frames_per_segment = int(segment_duration * fps)
        
        while True:
            segment_start_time = segment_number * segment_duration
            segment_end_time = min((segment_number + 1) * segment_duration, total_duration)
            
            if segment_start_time >= total_duration:
                break
            
            print(f"\n🔍 Analyzing segment {segment_number + 1}: {segment_start_time}s - {segment_end_time}s")
            
            # Jump to segment start
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment_number * frames_per_segment)
            
            segment_detections = []
            frames_analyzed = 0
            max_frames_in_segment = min(frames_per_segment, total_frames - segment_number * frames_per_segment)
            
            # Sample frames within segment (every 2 seconds for efficiency)
            frame_interval = int(2 * fps)  # Sample every 2 seconds
            
            for frame_idx in range(0, max_frames_in_segment, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, segment_number * frames_per_segment + frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                detection_result = self.detect_bombus_in_frame(frame)
                current_time = segment_start_time + (frame_idx / fps)
                
                segment_detections.append({
                    'timestamp': current_time,
                    'bee_count': detection_result['count'],
                    'max_confidence': max(detection_result['scores']) if len(detection_result['scores']) > 0 else 0.0,
                    'frame_number': segment_number * frames_per_segment + frame_idx
                })
                
                frames_analyzed += 1
            
            # Aggregate segment results
            if segment_detections:
                total_bees_detected = sum([d['bee_count'] for d in segment_detections])
                max_simultaneous_bees = max([d['bee_count'] for d in segment_detections])
                frames_with_bees = sum([1 for d in segment_detections if d['bee_count'] > 0])
                avg_confidence = np.mean([d['max_confidence'] for d in segment_detections if d['max_confidence'] > 0])
                
                results.append({
                    'segment_number': segment_number + 1,
                    'start_time': segment_start_time,
                    'end_time': segment_end_time,
                    'duration': segment_end_time - segment_start_time,
                    'total_bee_detections': total_bees_detected,
                    'max_simultaneous_bees': max_simultaneous_bees,
                    'frames_analyzed': frames_analyzed,
                    'frames_with_bees': frames_with_bees,
                    'detection_rate': frames_with_bees / frames_analyzed if frames_analyzed > 0 else 0,
                    'avg_confidence': avg_confidence if not np.isnan(avg_confidence) else 0.0,
                    'bee_activity_score': (frames_with_bees / frames_analyzed) * max_simultaneous_bees if frames_analyzed > 0 else 0
                })
                
                print(f"   🐝 Total detections: {total_bees_detected}")
                print(f"   🔢 Max simultaneous bees: {max_simultaneous_bees}")
                print(f"   📈 Detection rate: {frames_with_bees/frames_analyzed*100:.1f}%")
            
            segment_number += 1
        
        cap.release()
        return pd.DataFrame(results)

    def create_detection_visualization(self, results_df: pd.DataFrame, output_path: str):
        """
        Create visualization of bee activity across video segments
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Bombus Detection Analysis by Video Segment', fontsize=16)
        
        # 1. Total detections per segment
        axes[0,0].bar(results_df['segment_number'], results_df['total_bee_detections'], 
                     color='gold', alpha=0.7)
        axes[0,0].set_title('Total Bee Detections per Segment')
        axes[0,0].set_xlabel('Segment Number')
        axes[0,0].set_ylabel('Total Detections')
        
        # 2. Max simultaneous bees
        axes[0,1].plot(results_df['segment_number'], results_df['max_simultaneous_bees'], 
                      marker='o', linewidth=2, markersize=6, color='orange')
        axes[0,1].set_title('Maximum Simultaneous Bees')
        axes[0,1].set_xlabel('Segment Number')
        axes[0,1].set_ylabel('Max Bees at Once')
        
        # 3. Detection rate heatmap by time
        axes[1,0].bar(results_df['segment_number'], results_df['detection_rate'], 
                     color='lightblue', alpha=0.8)
        axes[1,0].set_title('Detection Rate by Segment')
        axes[1,0].set_xlabel('Segment Number')
        axes[1,0].set_ylabel('Fraction of Frames with Bees')
        
        # 4. Activity score timeline
        axes[1,1].fill_between(results_df['segment_number'], results_df['bee_activity_score'], 
                              alpha=0.6, color='green')
        axes[1,1].set_title('Bee Activity Score Timeline')
        axes[1,1].set_xlabel('Segment Number')
        axes[1,1].set_ylabel('Activity Score')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()


class VideoBatchProcessor:
    """
    Process multiple videos and generate comprehensive reports
    """
    
    def __init__(self, detector: BombusObjectDetector):
        self.detector = detector
        
    def process_video_directory(self, video_dir: str, output_dir: str, segment_duration: int = 300):
        """
        Process all videos in directory and generate reports
        """
        os.makedirs(output_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        all_results = []
        
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            print(f"\n{'='*50}")
            print(f"Processing: {video_file}")
            
            # Analyze video
            results_df = self.detector.analyze_video_segments(video_path, segment_duration)
            results_df['video_name'] = video_file
            
            # Save individual video results
            video_output_path = os.path.join(output_dir, f"{video_file.replace('.mp4', '')}_segments.csv")
            results_df.to_csv(video_output_path, index=False)
            
            # Create visualization
            viz_path = os.path.join(output_dir, f"{video_file.replace('.mp4', '')}_analysis.png")
            self.detector.create_detection_visualization(results_df, viz_path)
            
            all_results.append(results_df)
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, 'all_videos_segment_analysis.csv'), index=False)
        
        # Generate summary report
        self.generate_summary_report(combined_df, output_dir)
        
        return combined_df
    
    def generate_summary_report(self, combined_df: pd.DataFrame, output_dir: str):
        """
        Generate comprehensive summary report
        """
        summary = {
            'total_videos_processed': combined_df['video_name'].nunique(),
            'total_segments_analyzed': len(combined_df),
            'total_bee_detections': combined_df['total_bee_detections'].sum(),
            'avg_bees_per_video': combined_df.groupby('video_name')['total_bee_detections'].sum().mean(),
            'max_simultaneous_bees_overall': combined_df['max_simultaneous_bees'].max(),
            'videos_with_bee_activity': (combined_df.groupby('video_name')['total_bee_detections'].sum() > 0).sum(),
            'avg_detection_rate': combined_df['detection_rate'].mean(),
            'peak_activity_segments': combined_df.nlargest(5, 'bee_activity_score')[['video_name', 'segment_number', 'bee_activity_score']]
        }
        
        # Save summary
        with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📊 ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Videos processed: {summary['total_videos_processed']}")
        print(f"Total bee detections: {summary['total_bee_detections']}")
        print(f"Average bees per video: {summary['avg_bees_per_video']:.1f}")
        print(f"Max simultaneous bees: {summary['max_simultaneous_bees_overall']}")
        print(f"Videos with activity: {summary['videos_with_bee_activity']}")


# Usage Example
if __name__ == "__main__":
    # Initialize detector
    detector = BombusObjectDetector(
        model_path='models/bombus_object_detection_model.pth',
        confidence_threshold=0.5
    )
    
    # Process single video with 5-minute segments
    video_path = "/path/to/your/video.mp4"
    results = detector.analyze_video_segments(video_path, segment_duration=300)
    
    # Create visualization
    detector.create_detection_visualization(results, 'bee_activity_analysis.png')
    
    # Process entire directory
    processor = VideoBatchProcessor(detector)
    all_results = processor.process_video_directory(
        video_dir="/path/to/video/directory",
        output_dir="analysis_results",
        segment_duration=300  # 5 minutes
    )
    
    print("\n✅ Analysis complete! Check analysis_results/ for detailed outputs.")