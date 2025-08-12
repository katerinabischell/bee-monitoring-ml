#!/usr/bin/env python3
"""
Create object detection dataset using Ventura Milk Vetch video P1000471.MP4
Based on manual annotations with quadrant-based bee locations
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime
import yaml
import re

class VenturaP1000471Dataset:
    """
    Create YOLO dataset from Ventura Milk Vetch video P1000471.MP4 with quadrant-based bee tracking
    
    Quadrant system:
    1 = top left       2 = top middle      3 = top right
    4 = middle left    5 = middle middle   6 = middle right  
    7 = bottom left    8 = bottom middle   9 = bottom right
    """
    
    def __init__(self, output_dir="data"):
        # Create structured output directory
        self.base_dir = Path(output_dir)
        self.extracted_frames_dir = self.base_dir / "extracted_frames"
        self.annotations_dir = self.base_dir / "annotations"
        self.setup_directory_structure()
        
        # Define quadrant centers for bounding box placement
        self.quadrant_centers = {
            1: (0.25, 0.25),  # top left
            2: (0.50, 0.25),  # top middle  
            3: (0.75, 0.25),  # top right
            4: (0.25, 0.50),  # middle left
            5: (0.50, 0.50),  # middle middle
            6: (0.75, 0.50),  # middle right
            7: (0.25, 0.75),  # bottom left
            8: (0.50, 0.75),  # bottom middle
            9: (0.75, 0.75),  # bottom right
        }
        
    def setup_directory_structure(self):
        """Create organized directory structure for the full pipeline"""
        dirs = [
            self.base_dir / "raw_videos",
            self.base_dir / "extracted_frames", 
            self.base_dir / "annotations",
            self.base_dir / "models",
            self.base_dir / "results",
            self.base_dir / "config"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Created directory structure in {self.base_dir}")
    
    def parse_timestamp_to_seconds(self, timestamp_str):
        """Convert timestamp string (MM:SS) to seconds"""
        try:
            parts = timestamp_str.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        except:
            print(f"⚠️ Could not parse timestamp: {timestamp_str}")
            return None
    
    def parse_quadrants(self, quadrant_str):
        """Parse quadrant string into list of integers representing frame regions"""
        if not quadrant_str or quadrant_str.strip() == "":
            return []
        
        # Handle cases like "5, 5" or "7, 5, 3" 
        quadrants = []
        for quad in quadrant_str.split(','):
            quad = quad.strip()
            if quad.isdigit() and 1 <= int(quad) <= 9:
                quadrants.append(int(quad))
        
        return quadrants
    
    def get_quadrant_name(self, quadrant_num):
        """Get human-readable name for quadrant"""
        quadrant_names = {
            1: "top-left", 2: "top-middle", 3: "top-right",
            4: "middle-left", 5: "middle-center", 6: "middle-right",
            7: "bottom-left", 8: "bottom-middle", 9: "bottom-right"
        }
        return quadrant_names.get(quadrant_num, f"unknown-{quadrant_num}")
    
    def prepare_ventura_p1000471_dataset(self, manual_annotations):
        """
        Prepare dataset using your manual annotations from P1000471.MP4
        
        Args:
            manual_annotations: List of manual annotations with timestamps and quadrant locations
        """
        print(f"\n{'='*70}")
        print(f"PREPARING VENTURA MILK VETCH P1000471 DATASET")
        print(f"Using Quadrant-Based Bee Location System")
        print(f"{'='*70}")
        
        # Ventura Milk Vetch video path - Week 6, Day 1, Site 1, Morning
        possible_video_paths = [
            "/Volumes/Expansion1/summer2025_ncos_kb_collections/ventura_milkvetch/week 6/day 1/site 1/morning/P1000471.MP4",
            "/Volumes/Expansion1/summer2025_ncos_kb_collections/ventura_milk_vetch/week 6/day 1/site 1/morning/P1000471.MP4",
            "/Volumes/Expansion1/summer2025_ncos_kb_collections/week 6/day 1/site 1/morning/P1000471.MP4",
            "/Volumes/Expansion1/P1000471.MP4",  # If moved to root
            "data/raw_videos/P1000471.MP4",  # If in our organized structure
            "P1000471.MP4",  # If in current directory
        ]
        
        video_path = None
        for path in possible_video_paths:
            if os.path.exists(path):
                video_path = path
                break
        
        if not video_path:
            print(f"❌ Video P1000471.MP4 not found in expected locations:")
            for path in possible_video_paths:
                print(f"   - {path}")
            print("💡 Please provide the correct path to P1000471.MP4")
            return None, None
        
        print(f"✅ Found video: {video_path}")
        
        # Extract frames
        extracted_frames = self.extract_frames_from_manual_annotations(
            video_path, manual_annotations, "P1000471_ventura"
        )
        
        # Create config
        config_path = self.create_yolo_config()
        
        # Generate summary - all frames have bees
        total_frames = len(extracted_frames)
        total_bees = sum([len(f['quadrants']) for f in extracted_frames])
        
        # Quadrant usage statistics
        quadrant_usage = {}
        for frame in extracted_frames:
            for quad in frame['quadrants']:
                quadrant_usage[quad] = quadrant_usage.get(quad, 0) + 1
        
        print(f"\n📊 VENTURA MILK VETCH P1000471 DATASET SUMMARY:")
        print(f"   Total frames with bees: {total_frames}")
        print(f"   Total bee instances: {total_bees}")
        print(f"   Average bees per frame: {total_bees/total_frames:.2f}")
        print(f"   Video length: 21 minutes")
        print(f"   Plant type: Ventura Milk Vetch")
        
        print(f"\n📍 QUADRANT USAGE STATISTICS:")
        for quad in sorted(quadrant_usage.keys()):
            quad_name = self.get_quadrant_name(quad)
            count = quadrant_usage[quad]
            print(f"   Quadrant {quad} ({quad_name}): {count} instances")
        
        return extracted_frames, config_path
    
    def extract_frames_from_manual_annotations(self, video_path, annotations, output_name_prefix):
        """Extract frames from video at manually annotated timestamps"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        extracted_frames = []
        
        print(f"🎬 Extracting frames from {Path(video_path).name}")
        print(f"   Total annotations: {len(annotations)}")
        print(f"   Video FPS: {fps:.2f}")
        
        for i, annotation in enumerate(annotations):
            timestamp_seconds = annotation['timestamp_seconds']
            quadrants = annotation['quadrants']
            has_bee = len(quadrants) > 0
            
            # Jump to timestamp
            frame_number = int(timestamp_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Could not extract frame at {timestamp_seconds}s")
                continue
            
            # Save frame
            frame_filename = f"{output_name_prefix}_t{int(timestamp_seconds):04d}.jpg"
            frame_path = self.extracted_frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Create initial YOLO annotation
            label_filename = f"{output_name_prefix}_t{int(timestamp_seconds):04d}.txt"
            label_path = self.annotations_dir / label_filename
            
            # All timestamps have bees (create bounding boxes based on quadrants)
            bbox_annotations = []
            
            for quad in quadrants:
                if quad in self.quadrant_centers:
                    # Get quadrant center coordinates
                    x_center, y_center = self.quadrant_centers[quad]
                    
                    # Default bounding box size (will be refined in annotation step)
                    width = 0.20   # 20% of frame width
                    height = 0.20  # 20% of frame height
                    
                    # Create YOLO format annotation (class x_center y_center width height)
                    bbox_annotation = f"0 {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}"
                    bbox_annotations.append(bbox_annotation)
            
            with open(label_path, 'w') as f:
                for bbox in bbox_annotations:
                    f.write(bbox + "\n")
            
            quadrants_str = ", ".join([f"{q}({self.get_quadrant_name(q)})" for q in quadrants])
            print(f"   ✅ Frame {i+1}: {timestamp_seconds}s -> {len(quadrants)} BEE(S) in quadrants: {quadrants_str}")
            
            extracted_frames.append({
                'timestamp': annotation['timestamp'],
                'timestamp_seconds': timestamp_seconds,
                'frame_filename': frame_filename,
                'label_filename': label_filename,
                'has_bee': True,  # All timestamps have bees
                'quadrants': quadrants,
                'bee_count': len(quadrants),
                'quadrant_names': [self.get_quadrant_name(q) for q in quadrants],
                'plant_type': 'ventura_milk_vetch',
                'video': 'P1000471',
                'needs_bbox_refinement': True  # All frames need refinement
            })
        
        cap.release()
        
        print(f"✅ Extracted {len(extracted_frames)} frames from P1000471.MP4")
        return extracted_frames
    
    def create_yolo_config(self):
        """Create YOLO configuration files"""
        dataset_config = {
            'path': str(self.base_dir.absolute()),
            'train': 'extracted_frames',  # Will be reorganized later
            'val': 'extracted_frames',    # Will be split later
            'test': 'extracted_frames',   # Will be split later
            'nc': 1,
            'names': ['bombus']
        }
        
        config_path = self.base_dir / "config" / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Created YOLO config: {config_path}")
        return config_path
    
    def save_annotation_summary(self, extracted_frames):
        """Save a summary of annotations for reference"""
        summary_data = []
        
        for frame in extracted_frames:
            summary_data.append({
                'timestamp': frame['timestamp'],
                'timestamp_seconds': frame['timestamp_seconds'],
                'frame_filename': frame['frame_filename'],
                'has_bee': frame['has_bee'],
                'bee_count': frame['bee_count'],
                'quadrants': ', '.join(map(str, frame['quadrants'])) if frame['quadrants'] else '',
                'quadrant_names': ', '.join(frame['quadrant_names']) if frame['quadrant_names'] else '',
                'needs_annotation_refinement': frame['needs_bbox_refinement']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.base_dir / "annotation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"✅ Saved annotation summary: {summary_path}")
        return summary_path
    
    def create_quadrant_reference(self):
        """Create a visual reference for the quadrant system"""
        reference_content = """
# Ventura Milkvetch P1000471 Quadrant Reference

## Quadrant System Layout:
```
┌─────────┬─────────┬─────────┐
│    1    │    2    │    3    │
│top-left │top-mid  │top-right│
├─────────┼─────────┼─────────┤
│    4    │    5    │    6    │
│mid-left │ center  │mid-right│
├─────────┼─────────┼─────────┤
│    7    │    8    │    9    │
│bot-left │bot-mid  │bot-right│
└─────────┴─────────┴─────────┘
```

## Coordinates Used for Bounding Boxes:
- Quadrant 1 (top-left): Center at (0.25, 0.25)
- Quadrant 2 (top-middle): Center at (0.50, 0.25) 
- Quadrant 3 (top-right): Center at (0.75, 0.25)
- Quadrant 4 (middle-left): Center at (0.25, 0.50)
- Quadrant 5 (middle-center): Center at (0.50, 0.50)
- Quadrant 6 (middle-right): Center at (0.75, 0.50)
- Quadrant 7 (bottom-left): Center at (0.25, 0.75)
- Quadrant 8 (bottom-middle): Center at (0.50, 0.75)
- Quadrant 9 (bottom-right): Center at (0.75, 0.75)

Default bounding box size: 20% x 20% of frame
"""
        
        reference_path = self.base_dir / "QUADRANT_REFERENCE.md"
        with open(reference_path, 'w') as f:
            f.write(reference_content)
        
        print(f"✅ Created quadrant reference: {reference_path}")
        return reference_path


def create_p1000471_manual_annotations():
    """
    Your actual manual annotations from P1000471.MP4
    Week 6 - Day 1 - Site 1 - Morning (21 min video)
    
    Quadrant system:
    1=top-left, 2=top-middle, 3=top-right
    4=middle-left, 5=middle-center, 6=middle-right  
    7=bottom-left, 8=bottom-middle, 9=bottom-right
    """
    
    # Your raw annotation data with quadrant locations
    raw_annotations = """
00:26 | YES | 7
01:44 | YES | 4
01:55 | YES | 4
02:03 | YES | 4
02:20 | YES | 4
02:26 | YES | 7
02:29 | YES | 7
02:44 | YES | 7
03:01 | YES | 7
03:09 | YES | 7
03:34 | 7
03:39 | 9
03:50 | 5
03:56 | 5
04:01 | 5
04:12 | 8
04:24 | 9
04:31 | 3
04:42 | 5
04:43 | 8
05:02 | 4
05:05 | 4
05:10 | 4
05:17 | 4
05:35 | 4
05:46 | 5
05:47 | 8, 4
05:53 | 4
05:59 | 4
06:04 | 7
06:06 | 7
06:29 | 9
06:30 | 9
06:34 | 5
06:35 | 5
06:36 | 5, 7
06:39 | 4, 7, 9
06:41 | 7
06:43 | 1
06:51 | 1
06:52 | 1, 2
06:60 | 8
07:01 | 8
07:04 | 6
07:08 | 6
07:12 | 9
07:17 | 9
07:29 | 9
07:33 | 2
07:36 | 2
08:06 | 2
08:51 | 4
09:43 | 1
09:48 | 1
10:06 | 1
10:34 | 1
10:55 | 5
11:21 | 5
11:22 | 5, 5
11:25 | 5, 5
11:30 | 5
11:52 | 6
11:55 | 5
14:01 | 7
14:05 | 7
14:12 | 7
14:34 | 8
14:38 | 8
14:39 | 8, 9
14:57 | 4
15:10 | 4
15:15 | 4
15:26 | 7
15:33 | 5
15:40 | 4
15:48 | 4, 5
16:04 | 1
16:06 | 1
16:12 | 1, 2
16:24 | 3
16:28 | 6
16:29 | 6
16:32 | 2
16:41 | 6
16:45 | 3
16:47 | 7, 5, 3
16:50 | 7, 5, 3
16:53 | 7, 5
16:58 | 7, 5
17:03 | 7, 5
17:06 | 7, 5, 6
17:09 | 7, 5, 6
17:13 | 7, 5, 6
17:18 | 7, 6
17:27 | 7, 6
17:31 | 5
17:33 | 8
17:37 | 8
17:42 | 8
17:49 | 8, 7
17:52 | 7, 7
18:07 | 7
18:12 | 7
18:18 | 4
18:20 | 4
18:23 | 4
18:28 | 4
18:31 | 4, 7
18:41 | 7, 8
18:42 | 7, 5
18:47 | 1, 5, 2
18:54 | 6
18:56 | 5, 6
19:02 | 9
19:05 | 9
19:12 | 3
19:17 | 3
19:21 | 6
19:24 | 6
19:35 | 5
19:40 | 2
19:54 | 3
20:09 | 3
20:26 | 2
20:32 | 7
21:02 | 7
"""
    
    manual_annotations = []
    dataset_creator = VenturaP1000471Dataset()
    
    for line in raw_annotations.strip().split('\n'):
        if not line.strip():
            continue
            
        parts = line.split('|')
        if len(parts) < 2:
            continue
        
        timestamp = parts[0].strip()
        
        # Get quadrant data (skip "YES" if present)
        if len(parts) == 3 and parts[1].strip() == "YES":
            quadrants_str = parts[2].strip()
        elif len(parts) == 2:
            quadrants_str = parts[1].strip()
        else:
            continue  # Skip malformed lines
        
        # Convert timestamp to seconds
        timestamp_seconds = dataset_creator.parse_timestamp_to_seconds(timestamp)
        if timestamp_seconds is None:
            continue
        
        # Parse quadrant locations
        quadrants = dataset_creator.parse_quadrants(quadrants_str)
        
        manual_annotations.append({
            'timestamp': timestamp,
            'timestamp_seconds': timestamp_seconds,
            'quadrants': quadrants,
            'has_bee': True  # All timestamps have bees
        })
    
    # Statistics - all frames have bees
    total_frames = len(manual_annotations)
    total_bee_instances = sum([len(a['quadrants']) for a in manual_annotations])
    
    # Quadrant usage analysis
    quadrant_usage = {}
    for annotation in manual_annotations:
        for quad in annotation['quadrants']:
            quadrant_usage[quad] = quadrant_usage.get(quad, 0) + 1
    
    print(f"📊 P1000471 MANUAL ANNOTATION SUMMARY:")
    print(f"   Video: Week 6, Day 1, Site 1, Morning (21 min)")
    print(f"   Total frames with bees: {total_frames}")
    print(f"   Total bee instances: {total_bee_instances}")
    print(f"   Average bees per frame: {total_bee_instances/total_frames:.2f}")
    print(f"   Most active quadrants: {sorted(quadrant_usage.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    return manual_annotations


def main():
    """Main function - create dataset from P1000471 manual annotations"""
    print(f"🌸 VENTURA MILK VETCH P1000471 OBJECT DETECTION DATASET")
    print(f"{'='*70}")
    
    print("🎬 Creating dataset from Week 6, Day 1, Site 1, Morning video...")
    print("📍 Using quadrant-based bee location system (1-9 grid)")
    
    # Get your manual annotations
    manual_annotations = create_p1000471_manual_annotations()
    
    # Create dataset
    dataset_creator = VenturaP1000471Dataset()
    extracted_frames, config_path = dataset_creator.prepare_ventura_p1000471_dataset(manual_annotations)
    
    if extracted_frames:
        # Save annotation summary and quadrant reference
        summary_path = dataset_creator.save_annotation_summary(extracted_frames)
        reference_path = dataset_creator.create_quadrant_reference()
        
        print(f"\n✅ DATASET CREATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Dataset location: {dataset_creator.base_dir}")
        print(f"Frames extracted: {len(extracted_frames)}")
        print(f"Total bee instances: {sum([f['bee_count'] for f in extracted_frames])}")
        print(f"Video: P1000471.MP4 (Week 6, Day 1, Site 1, Morning)")
        print(f"Plant type: Ventura Milk Vetch")
        
        print(f"\n📁 FOLDER STRUCTURE CREATED:")
        print(f"   📁 data/")
        print(f"      ├── 📁 raw_videos/          # Place original MP4s here")
        print(f"      ├── 📁 extracted_frames/    # {len(extracted_frames)} JPG frames")
        print(f"      ├── 📁 annotations/         # {len(extracted_frames)} YOLO label files")
        print(f"      ├── 📁 models/              # For trained model weights")
        print(f"      ├── 📁 results/             # For test results")
        print(f"      └── 📁 config/              # YOLO configuration")
        
        print(f"\n📋 FILES CREATED:")
        print(f"   - Annotation summary: {summary_path}")
        print(f"   - Quadrant reference: {reference_path}")
        print(f"   - YOLO config: {config_path}")
        
        print(f"\n📍 QUADRANT SYSTEM:")
        print(f"   1=top-left    2=top-middle    3=top-right")
        print(f"   4=middle-left 5=center        6=middle-right")  
        print(f"   7=bottom-left 8=bottom-middle 9=bottom-right")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Create scripts/2_annotate_gui.py for bbox refinement")
        print(f"2. Create scripts/3_train_model.py for YOLOv8 training")
        print(f"3. Create scripts/4_test_videos.py for testing multiple videos")
        print(f"4. Move P1000471.MP4 to data/raw_videos/ folder")
        
        return True
    else:
        print("❌ Dataset creation failed")
        return False

if __name__ == "__main__":
    main()