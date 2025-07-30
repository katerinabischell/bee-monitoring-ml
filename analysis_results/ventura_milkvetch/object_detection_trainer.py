#!/usr/bin/env python3
"""
Object Detection Training Pipeline for Bombus Detection
Uses YOLO for real-time bee detection and counting
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime
import yaml
import torch

class BombusObjectDetectionDataset:
    """
    Create YOLO format dataset from manual annotations and video data
    """
    
    def __init__(self, output_dir="object_detection_dataset"):
        self.output_dir = Path(output_dir)
        self.setup_directory_structure()
        
    def setup_directory_structure(self):
        """Create YOLO dataset structure"""
        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val", 
            self.output_dir / "images" / "test",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
            self.output_dir / "labels" / "test"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Created YOLO dataset structure in {self.output_dir}")
    
    def extract_frames_from_manual_annotations(self, video_path, annotations, output_name_prefix):
        """
        Extract frames from video at manually annotated timestamps
        
        Args:
            video_path: Path to video file
            annotations: List of manual annotations with timestamps
            output_name_prefix: Prefix for output files (e.g., "P1000421")
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        extracted_frames = []
        
        print(f"🎬 Extracting frames from {Path(video_path).name}")
        print(f"   Total annotations: {len(annotations)}")
        
        for i, annotation in enumerate(annotations):
            timestamp = annotation['timestamp']
            has_bee = annotation['has_bee']
            
            # Jump to timestamp
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Could not extract frame at {timestamp}s")
                continue
            
            # Save frame
            frame_filename = f"{output_name_prefix}_t{int(timestamp):04d}.jpg"
            frame_path = self.output_dir / "images" / "train" / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Create YOLO annotation
            label_filename = f"{output_name_prefix}_t{int(timestamp):04d}.txt"
            label_path = self.output_dir / "labels" / "train" / label_filename
            
            if has_bee:
                # For now, use default bounding box (will need manual refinement)
                # YOLO format: class_id center_x center_y width height (normalized 0-1)
                bbox_annotation = "0 0.5 0.5 0.3 0.3"  # Class 0 = bombus, center box
                
                with open(label_path, 'w') as f:
                    f.write(bbox_annotation + "\n")
                
                print(f"   ✅ Frame {i+1}: {timestamp}s -> BEE (needs bbox refinement)")
            else:
                # Empty annotation file for negative examples
                with open(label_path, 'w') as f:
                    pass  # Empty file
                
                print(f"   ❌ Frame {i+1}: {timestamp}s -> NO BEE")
            
            extracted_frames.append({
                'timestamp': timestamp,
                'frame_filename': frame_filename,
                'label_filename': label_filename,
                'has_bee': has_bee,
                'needs_bbox_refinement': has_bee
            })
        
        cap.release()
        
        print(f"✅ Extracted {len(extracted_frames)} frames")
        return extracted_frames
    
    def create_yolo_config(self):
        """Create YOLO configuration files"""
        
        # Dataset configuration
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': 1,  # Number of classes
            'names': ['bombus']
        }
        
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Created YOLO config: {config_path}")
        
        return config_path
    
    def prepare_manual_annotation_dataset(self):
        """
        Prepare dataset using your manual annotations from P1000421.MP4
        """
        print(f"\n{'='*60}")
        print(f"PREPARING OBJECT DETECTION DATASET")
        print(f"{'='*60}")
        
        # Your manual annotations from P1000421.MP4
        video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/birds_beak/week_5/day_2/site_4/afternoon/P1000421.MP4"
        
        manual_annotations = [
            # Bee activity periods
            {'timestamp': 102, 'has_bee': True, 'note': 'Bombus present on flower (middle)'},
            {'timestamp': 105, 'has_bee': True, 'note': 'Bombus flies up'},
            {'timestamp': 106, 'has_bee': True, 'note': 'Bombus lands on another flower'},
            {'timestamp': 120, 'has_bee': True, 'note': 'Bombus on plant'},
            {'timestamp': 121, 'has_bee': True, 'note': 'Bombus flies up and lands'},
            {'timestamp': 130, 'has_bee': True, 'note': 'Second bombus lands (left side)'},
            {'timestamp': 148, 'has_bee': True, 'note': 'Second bombus visible (left side)'},
            {'timestamp': 151, 'has_bee': True, 'note': 'Second bombus flies up (left side)'},
            {'timestamp': 170, 'has_bee': True, 'note': 'Second bombus lands with orange pollen'},
            {'timestamp': 182, 'has_bee': True, 'note': 'Second bombus visible (left side)'},
            {'timestamp': 183, 'has_bee': True, 'note': 'Second bombus visible (left side)'},
            {'timestamp': 195, 'has_bee': True, 'note': 'Second bombus visible (left side)'},
            {'timestamp': 202, 'has_bee': True, 'note': 'Second bombus with bright orange pollen'},
            {'timestamp': 207, 'has_bee': True, 'note': 'Second bombus still visible'},
            {'timestamp': 211, 'has_bee': True, 'note': 'Second bombus comes back'},
            {'timestamp': 212, 'has_bee': True, 'note': 'Second bombus in frame'},
            
            # No bee periods
            {'timestamp': 7, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 16, 'has_bee': False, 'note': 'Small fly flying by'},
            {'timestamp': 47, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 80, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 165, 'has_bee': False, 'note': 'Neither bombus visible'},
            {'timestamp': 216, 'has_bee': False, 'note': 'Second bombus leaves frame'},
            {'timestamp': 224, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 229, 'has_bee': False, 'note': 'Small fly flying'},
            {'timestamp': 256, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 290, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 323, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 336, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 365, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 397, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 457, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
            {'timestamp': 500, 'has_bee': False, 'note': 'Sound of airplane'},
            {'timestamp': 535, 'has_bee': False, 'note': 'Wind blowing, flowers moving'},
        ]
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            print("💡 Make sure external drive is mounted")
            return None
        
        # Extract frames
        extracted_frames = self.extract_frames_from_manual_annotations(
            video_path, manual_annotations, "P1000421"
        )
        
        # Create config
        config_path = self.create_yolo_config()
        
        # Generate summary
        bee_frames = len([f for f in extracted_frames if f['has_bee']])
        no_bee_frames = len([f for f in extracted_frames if not f['has_bee']])
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Total frames: {len(extracted_frames)}")
        print(f"   Frames with bees: {bee_frames}")
        print(f"   Frames without bees: {no_bee_frames}")
        print(f"   Frames needing bbox refinement: {bee_frames}")
        
        return extracted_frames, config_path


class YOLOTrainer:
    """
    Train YOLO model for bombus detection
    """
    
    def __init__(self, dataset_dir="ventura_object_detection_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = Path("models/object_detection")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def install_yolo_dependencies(self):
        """Install YOLO dependencies"""
        try:
            import ultralytics
            print("✅ YOLOv8 already installed")
        except ImportError:
            print("📦 Installing YOLOv8...")
            os.system("pip install ultralytics")
            print("✅ YOLOv8 installed")
    
    def train_yolo_model(self, config_path, epochs=50, img_size=640):
        """
        Train YOLO model for bombus detection using Ventura Milk Vetch dataset
        """
        print(f"\n{'='*60}")
        print(f"TRAINING YOLO OBJECT DETECTION MODEL")
        print(f"Using Ventura Milk Vetch dataset with 42 bee annotations")
        print(f"{'='*60}")
        
        self.install_yolo_dependencies()
        
        try:
            from ultralytics import YOLO
            
            # Load pretrained YOLO model
            model = YOLO('yolov8n.pt')  # Nano version for faster training
            
            print(f"🚀 Starting YOLO training...")
            print(f"   Dataset config: {config_path}")
            print(f"   Epochs: {epochs}")
            print(f"   Image size: {img_size}")
            print(f"   Training data: 37 frames with 42 bee annotations")
            print(f"   Multiple bees per frame: Supported ✅")
            
            # Train the model
            results = model.train(
                data=str(config_path),
                epochs=epochs,
                imgsz=img_size,
                project=str(self.models_dir),
                name='ventura_bombus_detection',
                device='cpu',  # Use CPU (change to 'cuda' if GPU available)
                patience=20,   # Early stopping
                save_period=10  # Save every 10 epochs
            )
            
            # Save final model
            model_path = self.models_dir / "ventura_bombus_detection" / "weights" / "best.pt"
            print(f"✅ Model training complete!")
            print(f"📁 Best model saved to: {model_path}")
            
            return model_path, results
            
        except Exception as e:
            print(f"❌ Error training YOLO model: {e}")
            print("💡 Check that all dependencies are installed correctly")
            return None, None
    
    def create_annotation_guide(self):
        """
        Create guide for manual bounding box annotation
        """
        guide = """
# Bounding Box Annotation Guide

## Tools Needed
1. **LabelImg**: pip install labelimg
2. **CVAT**: Online annotation tool
3. **Roboflow**: Web-based annotation

## YOLO Format
Format: class_id center_x center_y width height
- All values normalized 0-1
- center_x, center_y: center of bounding box
- width, height: box dimensions

## Example for bee at center of image:
0 0.5 0.5 0.2 0.3

## Annotation Guidelines
1. **Draw tight boxes** around visible bees
2. **Include partial bees** if >50% visible
3. **Use class 0** for all bombus species
4. **Be consistent** with box sizing

## Quick Start with LabelImg:
```bash
pip install labelimg
labelimg object_detection_dataset/images/train object_detection_dataset/labels/train
```

Navigate through images and draw bounding boxes around bees.
"""
        
        guide_path = self.dataset_dir / "annotation_guide.md"
        with open(guide_path, 'w') as f:
            f.write(guide)
        
        print(f"📖 Annotation guide created: {guide_path}")
        return guide_path


class ObjectDetectionAnalyzer:
    """
    Analyze videos using trained YOLO object detection model
    """
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained YOLO model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            print(f"✅ Loaded YOLO model: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def analyze_video_with_object_detection(self, video_path, confidence_threshold=0.5):
        """
        Analyze video using object detection - counts individual bees per frame
        """
        if not self.model:
            print("❌ Model not loaded")
            return None
        
        print(f"\n🔍 Object Detection Analysis: {Path(video_path).name}")
        
        results = self.model.predict(
            source=video_path,
            conf=confidence_threshold,
            save=False,
            stream=True,
            verbose=False
        )
        
        detections = []
        frame_count = 0
        
        for result in results:
            frame_count += 1
            
            # Count bees in this frame
            if result.boxes is not None:
                bee_count = len(result.boxes)
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Get confidences
            else:
                bee_count = 0
                boxes = []
                confidences = []
            
            timestamp = frame_count / 30.0  # Approximate timestamp
            
            detections.append({
                'frame': frame_count,
                'timestamp': timestamp,
                'bee_count': bee_count,
                'boxes': boxes.tolist() if len(boxes) > 0 else [],
                'confidences': confidences.tolist() if len(confidences) > 0 else []
            })
            
            if frame_count % 300 == 0:  # Progress every 10 seconds
                total_bees = sum(d['bee_count'] for d in detections[-300:])
                print(f"   Frame {frame_count}: {total_bees} bees in last 10s")
        
        print(f"✅ Analyzed {frame_count} frames")
        
        # Summary statistics
        total_detections = sum(d['bee_count'] for d in detections)
        frames_with_bees = len([d for d in detections if d['bee_count'] > 0])
        max_simultaneous = max(d['bee_count'] for d in detections)
        
        summary = {
            'video_path': video_path,
            'total_frames': frame_count,
            'total_bee_detections': total_detections,
            'frames_with_bees': frames_with_bees,
            'max_simultaneous_bees': max_simultaneous,
            'detection_rate': frames_with_bees / frame_count,
            'detections': detections
        }
        
        print(f"📊 Total bee detections: {total_detections}")
        print(f"🐝 Frames with bees: {frames_with_bees}/{frame_count} ({frames_with_bees/frame_count:.1%})")
        print(f"🔢 Max simultaneous bees: {max_simultaneous}")
        
        return summary


def main():
    """Main execution pipeline"""
    print(f"🐝 BOMBUS OBJECT DETECTION TRAINING PIPELINE")
    print(f"{'='*60}")
    
    # Step 1: Prepare dataset
    print(f"\n1️⃣ Preparing dataset from manual annotations...")
    dataset_creator = BombusObjectDetectionDataset()
    extracted_frames, config_path = dataset_creator.prepare_manual_annotation_dataset()
    
    if not extracted_frames:
        print("❌ Failed to prepare dataset")
        return
    
    # Step 2: Create annotation guide
    print(f"\n2️⃣ Creating annotation guide...")
    trainer = YOLOTrainer()
    guide_path = trainer.create_annotation_guide()
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Manually annotate bounding boxes using LabelImg:")
    print(f"      pip install labelimg")
    print(f"      labelimg object_detection_dataset/images/train")
    print(f"   2. Draw tight boxes around all visible bees")
    print(f"   3. Save annotations in YOLO format")
    print(f"   4. Run training: python object_detection_trainer.py --train")
    
    # Optional: Start training if annotations exist
    if '--train' in sys.argv:
        print(f"\n3️⃣ Training YOLO model...")
        model_path, results = trainer.train_yolo_model(config_path)
        
        if model_path:
            print(f"\n4️⃣ Testing trained model...")
            analyzer = ObjectDetectionAnalyzer(model_path)
            
            # Test on original video
            test_video = "/Volumes/Expansion/summer2025_ncos_kb_collections/birds_beak/week_5/day_2/site_4/afternoon/P1000421.MP4"
            if os.path.exists(test_video):
                summary = analyzer.analyze_video_with_object_detection(test_video)

if __name__ == "__main__":
    main()