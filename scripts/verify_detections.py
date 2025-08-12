#!/usr/bin/env python3
"""
Verify bee detections by showing zoomed-in views and confidence filtering
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DetectionVerifier:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.results_dir = self.data_dir / "results"
        self.verification_dir = self.results_dir / "verification"
        
        # Create output directory
        self.verification_dir.mkdir(parents=True, exist_ok=True)
        
        # Video directory
        self.video_base_dir = Path("/Volumes/Expansion1/summer2025_ncos_kb_collections/ventura_milkvetch/week 6/day 1/site 1")
        
        print(f"🔍 DETECTION VERIFIER")
        print(f"Output directory: {self.verification_dir}")
    
    def find_trained_model(self):
        """Find the best trained model"""
        model_path_v2 = self.models_dir / 'ventura_bee_detection_v2' / 'weights' / 'best.pt'
        model_path_v1 = self.models_dir / 'ventura_bee_detection' / 'weights' / 'best.pt'
        
        if model_path_v2.exists():
            return model_path_v2
        elif model_path_v1.exists():
            return model_path_v1
        else:
            return None
    
    def load_detection_metadata(self):
        """Load detection metadata from compilation"""
        metadata_path = self.results_dir / "compiled_detections" / "detection_metadata.csv"
        
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            print(f"📊 Loaded {len(df)} individual bee detections")
            return df
        else:
            print(f"❌ No detection metadata found")
            print("💡 Run 5_compile_detections.py first")
            return None
    
    def get_video_path(self, video_name, period):
        """Get full path to video file"""
        return self.video_base_dir / period / video_name
    
    def extract_and_analyze_detection(self, row, model):
        """Extract detection and create verification image"""
        video_name = row['video_name']
        period = row['period']
        frame_number = int(row['frame_number'])
        bee_id = row['bee_id']
        confidence = row['confidence']
        
        # Bounding box coordinates
        x1, y1, x2, y2 = row['box_x1'], row['box_y1'], row['box_x2'], row['box_y2']
        
        # Get video and extract frame
        video_path = self.get_video_path(video_name, period)
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract bounding box region with padding
        padding = 50
        h, w = frame_rgb.shape[:2]
        
        crop_x1 = max(0, int(x1) - padding)
        crop_y1 = max(0, int(y1) - padding)
        crop_x2 = min(w, int(x2) + padding)
        crop_y2 = min(h, int(y2) + padding)
        
        # Crop the region
        cropped_region = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Adjust bounding box coordinates for cropped image
        adjusted_x1 = int(x1) - crop_x1
        adjusted_y1 = int(y1) - crop_y1
        adjusted_x2 = int(x2) - crop_x1
        adjusted_y2 = int(y2) - crop_y1
        
        return {
            'full_frame': frame_rgb,
            'cropped_region': cropped_region,
            'original_bbox': (x1, y1, x2, y2),
            'adjusted_bbox': (adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2),
            'crop_coords': (crop_x1, crop_y1, crop_x2, crop_y2),
            'confidence': confidence,
            'video_name': video_name,
            'period': period,
            'timestamp': row['timestamp_formatted'],
            'bee_id': bee_id
        }
    
    def create_verification_image(self, detection_data, save_path):
        """Create side-by-side verification image"""
        full_frame = detection_data['full_frame']
        cropped_region = detection_data['cropped_region']
        x1, y1, x2, y2 = detection_data['original_bbox']
        adj_x1, adj_y1, adj_x2, adj_y2 = detection_data['adjusted_bbox']
        confidence = detection_data['confidence']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Full frame with bounding box
        ax1.imshow(full_frame)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.set_title(f"Full Frame - {detection_data['video_name']}\n{detection_data['period'].title()} - {detection_data['timestamp']}")
        ax1.axis('off')
        
        # Right: Zoomed crop with bounding box
        ax2.imshow(cropped_region)
        rect2 = patches.Rectangle((adj_x1, adj_y1), adj_x2-adj_x1, adj_y2-adj_y1, 
                                linewidth=3, edgecolor='red', facecolor='none')
        ax2.add_patch(rect2)
        
        # Color-code title by confidence
        if confidence >= 0.7:
            title_color = 'green'
            confidence_label = 'HIGH CONFIDENCE'
        elif confidence >= 0.4:
            title_color = 'orange'
            confidence_label = 'MEDIUM CONFIDENCE'
        else:
            title_color = 'red'
            confidence_label = 'LOW CONFIDENCE'
        
        ax2.set_title(f"Zoomed Detection (Bee {detection_data['bee_id']})\nConfidence: {confidence:.3f} - {confidence_label}", 
                     color=title_color, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def analyze_confidence_distribution(self, metadata_df):
        """Analyze confidence score distribution"""
        print(f"\n📊 CONFIDENCE ANALYSIS")
        print(f"{'='*50}")
        
        confidence_bins = [
            (0.0, 0.3, "Very Low"),
            (0.3, 0.5, "Low"), 
            (0.5, 0.7, "Medium"),
            (0.7, 0.9, "High"),
            (0.9, 1.0, "Very High")
        ]
        
        print(f"Confidence Distribution:")
        for min_conf, max_conf, label in confidence_bins:
            count = len(metadata_df[(metadata_df['confidence'] >= min_conf) & 
                                  (metadata_df['confidence'] < max_conf)])
            percentage = (count / len(metadata_df)) * 100
            print(f"  {label:10} ({min_conf:.1f}-{max_conf:.1f}): {count:3d} detections ({percentage:5.1f}%)")
        
        avg_confidence = metadata_df['confidence'].mean()
        median_confidence = metadata_df['confidence'].median()
        
        print(f"\nOverall Statistics:")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Median confidence:  {median_confidence:.3f}")
        print(f"  Min confidence:     {metadata_df['confidence'].min():.3f}")
        print(f"  Max confidence:     {metadata_df['confidence'].max():.3f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        low_confidence_count = len(metadata_df[metadata_df['confidence'] < 0.4])
        if low_confidence_count > 0:
            print(f"  • {low_confidence_count} detections have low confidence (<0.4) - likely false positives")
            print(f"  • Consider filtering to confidence ≥ 0.4 for more reliable results")
        
        reliable_count = len(metadata_df[metadata_df['confidence'] >= 0.5])
        print(f"  • {reliable_count} detections have medium+ confidence (≥0.5) - more reliable")
        
        return {
            'avg_confidence': avg_confidence,
            'median_confidence': median_confidence,
            'low_confidence_count': low_confidence_count,
            'reliable_count': reliable_count
        }
    
    def filter_and_verify_detections(self, min_confidence=0.4, max_samples=20):
        """Filter detections by confidence and create verification images"""
        print(f"\n🔍 FILTERING AND VERIFYING DETECTIONS")
        print(f"{'='*60}")
        print(f"Minimum confidence: {min_confidence}")
        print(f"Max samples to verify: {max_samples}")
        
        # Load metadata
        metadata_df = self.load_detection_metadata()
        if metadata_df is None:
            return None
        
        # Analyze confidence distribution
        conf_analysis = self.analyze_confidence_distribution(metadata_df)
        
        # Filter by confidence
        filtered_df = metadata_df[metadata_df['confidence'] >= min_confidence].copy()
        print(f"\n📋 After filtering (confidence ≥ {min_confidence}):")
        print(f"   Remaining detections: {len(filtered_df)} of {len(metadata_df)} original")
        
        if len(filtered_df) == 0:
            print("❌ No detections meet confidence threshold")
            return None
        
        # Sort by confidence (highest first) and take sample
        sample_df = filtered_df.nlargest(max_samples, 'confidence')
        
        # Load model
        model_path = self.find_trained_model()
        if not model_path:
            return None
        
        model = YOLO(model_path)
        
        print(f"\n🖼️ Creating verification images for top {len(sample_df)} detections...")
        
        verification_files = []
        
        for idx, (_, row) in enumerate(sample_df.iterrows()):
            print(f"Processing {idx+1}/{len(sample_df)}: {row['video_name']} at {row['timestamp_formatted']} (conf: {row['confidence']:.3f})")
            
            # Extract detection data
            detection_data = self.extract_and_analyze_detection(row, model)
            if detection_data is None:
                continue
            
            # Create verification image
            safe_video_name = Path(row['video_name']).stem
            output_filename = f"verify_{safe_video_name}_{row['period']}_t{row['timestamp_formatted'].replace(':', 'm')}_bee{row['bee_id']}_conf{row['confidence']:.3f}.jpg"
            output_path = self.verification_dir / output_filename
            
            self.create_verification_image(detection_data, output_path)
            verification_files.append(output_path)
            
            print(f"   ✅ Saved: {output_filename}")
        
        # Create filtered results summary
        summary_path = self.verification_dir / "verification_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("BEE DETECTION VERIFICATION SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Original detections: {len(metadata_df)}\n")
            f.write(f"Filtered detections (conf ≥ {min_confidence}): {len(filtered_df)}\n")
            f.write(f"Verification images created: {len(verification_files)}\n")
            f.write(f"Average confidence: {conf_analysis['avg_confidence']:.3f}\n")
            f.write(f"Reliable detections (conf ≥ 0.5): {conf_analysis['reliable_count']}\n")
            f.write(f"\nRecommendation: Use confidence threshold ≥ 0.4 for reliable results\n")
        
        print(f"\n✅ VERIFICATION COMPLETE!")
        print(f"{'='*60}")
        print(f"📁 Verification images: {len(verification_files)} files")
        print(f"📄 Summary: {summary_path}")
        print(f"📍 Location: {self.verification_dir}")
        
        return {
            'verification_files': verification_files,
            'filtered_detections': len(filtered_df),
            'original_detections': len(metadata_df),
            'confidence_analysis': conf_analysis,
            'summary_path': summary_path
        }


def main():
    """Run detection verification"""
    # Check requirements
    if not Path("data/results/compiled_detections/detection_metadata.csv").exists():
        print("❌ No detection metadata found")
        print("💡 Run 5_compile_detections.py first")
        return
    
    verifier = DetectionVerifier()
    
    # Run verification with filtering
    results = verifier.filter_and_verify_detections(
        min_confidence=0.3,  # Lower threshold to see more examples
        max_samples=25       # Check up to 25 best detections
    )
    
    if results:
        print(f"\n🎉 Check {verifier.verification_dir} to see zoomed-in views of detections!")
        print(f"💡 Look for actual bee shapes, fuzzy bodies, wings, etc. in the red boxes")

if __name__ == "__main__":
    main()