#!/usr/bin/env python3
"""
Compile all video frames where bees were detected with bounding boxes
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BeeDetectionCompiler:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.results_dir = self.data_dir / "results"
        self.detections_dir = self.results_dir / "compiled_detections"
        
        # Create output directories
        self.detections_dir.mkdir(parents=True, exist_ok=True)
        
        # Video directory
        self.video_base_dir = Path("/Volumes/Expansion1/summer2025_ncos_kb_collections/ventura_milkvetch/week 6/day 1/site 1")
        
        print(f"🐝 BEE DETECTION COMPILER")
        print(f"Output directory: {self.detections_dir}")
        print(f"Video source: {self.video_base_dir}")
    
    def find_trained_model(self):
        """Find the best trained model"""
        model_path_v2 = self.models_dir / 'ventura_bee_detection_v2' / 'weights' / 'best.pt'
        model_path_v1 = self.models_dir / 'ventura_bee_detection' / 'weights' / 'best.pt'
        
        if model_path_v2.exists():
            print(f"✅ Using improved model: {model_path_v2}")
            return model_path_v2
        elif model_path_v1.exists():
            print(f"✅ Using original model: {model_path_v1}")
            return model_path_v1
        else:
            print(f"❌ No trained model found")
            return None
    
    def load_detection_results(self):
        """Load detection results from previous analysis"""
        csv_path = self.results_dir / "video_analysis" / "detection_results.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Filter only frames with bee detections
            bee_detections = df[df['num_bees'] > 0].copy()
            print(f"📊 Found {len(bee_detections)} frames with bee detections")
            return bee_detections
        else:
            print(f"❌ No detection results found at {csv_path}")
            print("💡 Run 4_test_videos.py first")
            return None
    
    def get_video_path(self, video_name, period):
        """Get full path to video file"""
        video_path = self.video_base_dir / period / video_name
        if video_path.exists():
            return video_path
        else:
            print(f"⚠️ Video not found: {video_path}")
            return None
    
    def extract_detection_frame(self, video_path, frame_number, timestamp_seconds, model):
        """Extract a specific frame and run detection"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Jump to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            return None, None
        
        # Run detection on this frame
        results = model(frame, verbose=False)
        
        # Get detection info
        boxes = results[0].boxes
        detections = []
        
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                detections.append({
                    'box_id': i + 1,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': conf,
                    'width': x2 - x1,
                    'height': y2 - y1
                })
        
        cap.release()
        return frame, detections
    
    def create_annotated_image(self, frame, detections, video_name, period, timestamp, save_path):
        """Create annotated image with bounding boxes and labels"""
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(frame_rgb)
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
            width = x2 - x1
            height = y2 - y1
            conf = detection['confidence']
            
            # Draw rectangle
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add confidence label
            label = f"Bee {detection['box_id']}: {conf:.2f}"
            ax.text(x1, y1-10, label, color='red', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        # Add title with metadata
        title = f"{video_name} - {period.title()} - {timestamp} - {len(detections)} bee(s) detected"
        ax.set_title(title, fontsize=14, pad=20)
        ax.axis('off')
        
        # Save image
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_detection_summary_grid(self, detection_files):
        """Create a grid showing all detections"""
        print(f"🖼️ Creating detection summary grid...")
        
        # Limit to first 20 detections for grid (too many would be overwhelming)
        sample_files = detection_files[:20]
        
        # Calculate grid size
        cols = 4
        rows = (len(sample_files) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        fig.suptitle(f'Bee Detections from Video Analysis\nShowing {len(sample_files)} of {len(detection_files)} total detections', 
                     fontsize=16)
        
        # Handle single row case
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, detection_file in enumerate(sample_files):
            row = idx // cols
            col = idx % cols
            
            # Load and display image
            img = plt.imread(detection_file)
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(detection_file.stem, fontsize=8)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(len(sample_files), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save grid
        grid_path = self.detections_dir / "detection_summary_grid.jpg"
        plt.savefig(grid_path, dpi=200, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved detection grid: {grid_path}")
        return grid_path
    
    def create_detection_metadata(self, all_detection_data):
        """Create comprehensive metadata file"""
        print(f"📄 Creating detection metadata...")
        
        # Create DataFrame with all detection details
        metadata_rows = []
        
        for data in all_detection_data:
            base_info = {
                'video_name': data['video_name'],
                'period': data['period'],
                'timestamp_formatted': data['timestamp_formatted'],
                'timestamp_seconds': data['timestamp_seconds'],
                'frame_number': data['frame_number'],
                'total_bees': len(data['detections']),
                'image_file': data['image_file'].name
            }
            
            # Add each detection as a separate row
            for detection in data['detections']:
                row = base_info.copy()
                row.update({
                    'bee_id': detection['box_id'],
                    'confidence': detection['confidence'],
                    'box_x1': detection['x1'],
                    'box_y1': detection['y1'],
                    'box_x2': detection['x2'],
                    'box_y2': detection['y2'],
                    'box_width': detection['width'],
                    'box_height': detection['height']
                })
                metadata_rows.append(row)
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_path = self.detections_dir / "detection_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        # Create summary statistics
        summary_stats = {
            'total_detection_frames': len(all_detection_data),
            'total_individual_bees': len(metadata_rows),
            'videos_analyzed': metadata_df['video_name'].nunique(),
            'time_periods': list(metadata_df['period'].unique()),
            'avg_confidence': metadata_df['confidence'].mean(),
            'confidence_range': [metadata_df['confidence'].min(), metadata_df['confidence'].max()],
            'avg_bees_per_frame': metadata_df.groupby(['video_name', 'frame_number'])['bee_id'].count().mean(),
            'max_bees_single_frame': metadata_df.groupby(['video_name', 'frame_number'])['bee_id'].count().max()
        }
        
        # Save summary
        summary_path = self.detections_dir / "detection_summary_stats.txt"
        with open(summary_path, 'w') as f:
            f.write("BEE DETECTION COMPILATION SUMMARY\n")
            f.write("="*50 + "\n\n")
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✅ Saved metadata: {metadata_path}")
        print(f"✅ Saved summary: {summary_path}")
        
        return metadata_path, summary_path, summary_stats
    
    def compile_all_detections(self, confidence_threshold=0.25):
        """Compile all video frames where bees were detected"""
        print(f"\n🐝 STARTING BEE DETECTION COMPILATION")
        print(f"{'='*60}")
        print(f"Confidence threshold: {confidence_threshold}")
        
        # Load detection results
        detection_results = self.load_detection_results()
        if detection_results is None:
            return None
        
        # Find and load model
        model_path = self.find_trained_model()
        if not model_path:
            return None
        
        model = YOLO(model_path)
        model.conf = confidence_threshold
        
        print(f"\n📋 Processing {len(detection_results)} frames with bee detections...")
        
        compiled_files = []
        all_detection_data = []
        
        for idx, row in detection_results.iterrows():
            video_name = row['video_name']
            period = row['period']
            frame_number = int(row['frame_number'])
            timestamp_seconds = row['timestamp_seconds']
            timestamp_formatted = row['timestamp_formatted']
            
            print(f"Processing {idx+1}/{len(detection_results)}: {video_name} at {timestamp_formatted}")
            
            # Get video path
            video_path = self.get_video_path(video_name, period)
            if not video_path:
                continue
            
            # Extract frame and run detection
            frame, detections = self.extract_detection_frame(
                video_path, frame_number, timestamp_seconds, model
            )
            
            if frame is not None and detections:
                # Create output filename
                safe_video_name = Path(video_name).stem
                output_filename = f"{safe_video_name}_{period}_t{timestamp_formatted.replace(':', 'm')}_bees{len(detections)}.jpg"
                output_path = self.detections_dir / output_filename
                
                # Create annotated image
                self.create_annotated_image(
                    frame, detections, video_name, period, 
                    timestamp_formatted, output_path
                )
                
                compiled_files.append(output_path)
                
                # Store metadata
                all_detection_data.append({
                    'video_name': video_name,
                    'period': period,
                    'frame_number': frame_number,
                    'timestamp_seconds': timestamp_seconds,
                    'timestamp_formatted': timestamp_formatted,
                    'detections': detections,
                    'image_file': output_path
                })
                
                print(f"   ✅ Saved: {output_filename} ({len(detections)} bees)")
            else:
                print(f"   ⚠️ No detections found (below threshold)")
        
        if compiled_files:
            # Create summary grid
            grid_path = self.create_detection_summary_grid(compiled_files)
            
            # Create metadata
            metadata_path, summary_path, stats = self.create_detection_metadata(all_detection_data)
            
            print(f"\n✅ COMPILATION COMPLETE!")
            print(f"{'='*60}")
            print(f"📁 Output directory: {self.detections_dir}")
            print(f"🖼️ Individual images: {len(compiled_files)} files")
            print(f"📊 Summary grid: {grid_path}")
            print(f"📄 Metadata CSV: {metadata_path}")
            print(f"📈 Summary stats: {summary_path}")
            print(f"\n📊 QUICK STATS:")
            print(f"   Total detection frames: {stats['total_detection_frames']}")
            print(f"   Individual bees detected: {stats['total_individual_bees']}")
            print(f"   Average confidence: {stats['avg_confidence']:.3f}")
            print(f"   Max bees in single frame: {stats['max_bees_single_frame']}")
            
            return {
                'compiled_files': compiled_files,
                'grid_path': grid_path,
                'metadata_path': metadata_path,
                'summary_path': summary_path,
                'stats': stats
            }
        else:
            print("❌ No detections compiled")
            return None


def main():
    """Run detection compilation"""
    # Check requirements
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed")
        return
    
    # Check if video analysis was run
    results_dir = Path("data/results/video_analysis")
    if not (results_dir / "detection_results.csv").exists():
        print("❌ No video analysis results found")
        print("💡 Run 4_test_videos.py first")
        return
    
    # Start compilation
    compiler = BeeDetectionCompiler()
    
    results = compiler.compile_all_detections(
        confidence_threshold=0.25  # Same as video analysis
    )
    
    if results:
        print(f"\n🎉 All bee detections compiled successfully!")
        print(f"Check {compiler.detections_dir} for all images and metadata")

if __name__ == "__main__":
    main()