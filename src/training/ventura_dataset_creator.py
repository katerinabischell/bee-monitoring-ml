#!/usr/bin/env python3
"""
Create object detection dataset using Ventura Milk Vetch video
Better bee visibility for training
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime
import yaml

class VenturaMilkVetchDataset:
    """
    Create YOLO dataset from Ventura Milk Vetch video with better bee visibility
    """
    
    def __init__(self, output_dir="ventura_object_detection_dataset"):
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
    
    def prepare_ventura_dataset(self, manual_annotations):
        """
        Prepare dataset using your manual annotations from P1000446.MP4
        
        Args:
            manual_annotations: List of manual annotations with timestamps
        """
        print(f"\n{'='*60}")
        print(f"PREPARING VENTURA MILK VETCH DATASET")
        print(f"{'='*60}")
        
        # Ventura Milk Vetch video path - CORRECT PATH
        video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milkvetch/week 5/day 1/site 1/afternoon/P1000446.MP4"
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            print("💡 Please provide the correct path to P1000446.MP4")
            
            # Try alternative paths
            possible_paths = [
                "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milkvetch/week 5/day 1/site 1/afternoon/P1000446.MP4",
                "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milk_vetch/week 5/day 1/site 1/afternoon/P1000446.MP4",
                "/Volumes/Expansion/summer2025_ncos_kb_collections/week 5/day 1/site 1/afternoon/P1000446.MP4",
            ]
            
            for alt_path in possible_paths:
                if os.path.exists(alt_path):
                    video_path = alt_path
                    print(f"✅ Found video at: {video_path}")
                    break
            else:
                return None
        
        # Extract frames
        extracted_frames = self.extract_frames_from_manual_annotations(
            video_path, manual_annotations, "P1000446_ventura"
        )
        
        # Create config
        config_path = self.create_yolo_config()
        
        # Generate summary
        bee_frames = len([f for f in extracted_frames if f['has_bee']])
        no_bee_frames = len([f for f in extracted_frames if not f['has_bee']])
        
        print(f"\n📊 VENTURA MILK VETCH DATASET SUMMARY:")
        print(f"   Total frames: {len(extracted_frames)}")
        print(f"   Frames with bees: {bee_frames}")
        print(f"   Frames without bees: {no_bee_frames}")
        print(f"   Plant type: Ventura Milk Vetch (better bee visibility)")
        
        return extracted_frames, config_path
    
    def extract_frames_from_manual_annotations(self, video_path, annotations, output_name_prefix):
        """Extract frames from video at manually annotated timestamps"""
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
                # Default bounding box for manual refinement
                bbox_annotation = "0 0.5 0.5 0.25 0.25"  # Slightly larger for milk vetch
                
                with open(label_path, 'w') as f:
                    f.write(bbox_annotation + "\n")
                
                print(f"   ✅ Frame {i+1}: {timestamp}s -> BEE (needs bbox refinement)")
            else:
                # Empty annotation file for negative examples
                with open(label_path, 'w') as f:
                    pass
                
                print(f"   ❌ Frame {i+1}: {timestamp}s -> NO BEE")
            
            extracted_frames.append({
                'timestamp': timestamp,
                'frame_filename': frame_filename,
                'label_filename': label_filename,
                'has_bee': has_bee,
                'plant_type': 'ventura_milk_vetch',
                'needs_bbox_refinement': has_bee
            })
        
        cap.release()
        
        print(f"✅ Extracted {len(extracted_frames)} frames from Ventura Milk Vetch video")
        return extracted_frames
    
    def create_yolo_config(self):
        """Create YOLO configuration files"""
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': 1,
            'names': ['bombus']
        }
        
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Created YOLO config: {config_path}")
        return config_path


def create_manual_annotations():
    """
    Your actual manual annotations from P1000446.MP4
    Based on your careful observation
    """
    
    manual_annotations = [
        # Early activity burst (0-80s)
        {'timestamp': 2, 'has_bee': True, 'note': '1st bombus on the left'},
        {'timestamp': 3, 'has_bee': True, 'note': '1st bombus on the left'},
        {'timestamp': 4, 'has_bee': True, 'note': '2nd bombus flies in on the left'},
        {'timestamp': 7, 'has_bee': True, 'note': '2nd bombus visible clearly on the left'},
        {'timestamp': 10, 'has_bee': True, 'note': '2nd bombus flies over to 1st bombus'},
        {'timestamp': 15, 'has_bee': True, 'note': '1st and 2nd bombus visible both flying on the left'},
        {'timestamp': 16, 'has_bee': True, 'note': '2nd bombus visible, the 1st left the frame'},
        {'timestamp': 17, 'has_bee': True, 'note': '2nd bombus visible'},
        {'timestamp': 20, 'has_bee': False, 'note': 'Nothing is visible, wind'},
        {'timestamp': 27, 'has_bee': True, 'note': '3rd bombus flies into frame'},
        {'timestamp': 32, 'has_bee': False, 'note': 'Nothing visible'},
        {'timestamp': 36, 'has_bee': True, 'note': '3rd bombus flies into frame on the left'},
        {'timestamp': 38, 'has_bee': True, 'note': '3rd bombus flying in frame'},
        {'timestamp': 40, 'has_bee': False, 'note': '3rd bombus flies behind flower and is hidden'},
        {'timestamp': 54, 'has_bee': False, 'note': 'Yellow moth flies into frame'},
        {'timestamp': 57, 'has_bee': True, 'note': '3rd bombus is seen on one of the top flowers in middle'},
        {'timestamp': 62, 'has_bee': True, 'note': '3rd bombus is seen on top in middle again'},
        {'timestamp': 68, 'has_bee': True, 'note': '3rd bombus seen flying left'},
        {'timestamp': 69, 'has_bee': True, 'note': '3rd bombus lands on flower top left'},
        {'timestamp': 71, 'has_bee': True, 'note': '3rd bombus is seen clearly top left'},
        {'timestamp': 74, 'has_bee': False, 'note': '3rd bombus flies out of frame at top'},
        
        # Quiet period
        {'timestamp': 120, 'has_bee': False, 'note': 'Still nothing visible, just wind'},
        {'timestamp': 180, 'has_bee': False, 'note': 'Still nothing visible, just wind'},
        {'timestamp': 239, 'has_bee': False, 'note': 'Still nothing'},
        
        # Mid-video activity (240-300s)
        {'timestamp': 240, 'has_bee': True, 'note': '4th bombus comes into frame from bottom right'},
        {'timestamp': 242, 'has_bee': True, 'note': '4th bombus very visible bottom right'},
        {'timestamp': 245, 'has_bee': False, 'note': '4th bombus goes out of frame on right side'},
        {'timestamp': 292, 'has_bee': True, 'note': '5th bombus visible on top right'},
        {'timestamp': 295, 'has_bee': False, 'note': '5th bombus flies out of frame top'},
        
        # Peak activity period (315-390s) - Multiple bees
        {'timestamp': 315, 'has_bee': True, 'note': '6th bombus flies into frame top middle'},
        {'timestamp': 323, 'has_bee': True, 'note': '6th bombus seen clearly top middle'},
        {'timestamp': 325, 'has_bee': True, 'note': '7th bombus comes into frame (both 6 and 7 are visible)'},
        {'timestamp': 326, 'has_bee': True, 'note': '6th and 7th bombus clearly visible top middle'},
        {'timestamp': 336, 'has_bee': True, 'note': '6th and 7th bombus clearly visible top middle'},
        {'timestamp': 338, 'has_bee': True, 'note': '7th bombus flies out of frame at top, 6th still visible'},
        {'timestamp': 355, 'has_bee': True, 'note': '8th bombus flies into left side, 6th bombus still visible'},
        {'timestamp': 357, 'has_bee': True, 'note': '8th still visible on right, 6th gets covered by a flower'},
        {'timestamp': 364, 'has_bee': True, 'note': '8th bombus clearly visible on left'},
        {'timestamp': 369, 'has_bee': True, 'note': '8th bombus moved to top left'},
        {'timestamp': 380, 'has_bee': True, 'note': '8th bombus top left'},
        {'timestamp': 386, 'has_bee': True, 'note': '8th bombus top left'},
        {'timestamp': 389, 'has_bee': False, 'note': '8th bombus flies out of frame'},
        
        # Late activity (420-480s)
        {'timestamp': 420, 'has_bee': True, 'note': '9th bombus flies in top right'},
        {'timestamp': 422, 'has_bee': True, 'note': '9th bombus flies out at top'},
        {'timestamp': 424, 'has_bee': False, 'note': '9th bombus gone'},
        {'timestamp': 430, 'has_bee': True, 'note': '9th bombus flies back at top'},
        {'timestamp': 431, 'has_bee': False, 'note': '9th bombus gone'},
        {'timestamp': 433, 'has_bee': True, 'note': '9th bombus seen top middle'},
        {'timestamp': 435, 'has_bee': False, 'note': '9th bombus flies out top again'},
        {'timestamp': 456, 'has_bee': True, 'note': '10th bombus seen top right corner'},
        {'timestamp': 460, 'has_bee': False, 'note': '10th bombus flies out of frame'},
        {'timestamp': 471, 'has_bee': True, 'note': '11th bombus seen top left corner'},
        {'timestamp': 473, 'has_bee': False, 'note': '11th bombus leaves frame'},
        
        # End period - no activity
        {'timestamp': 480, 'has_bee': False, 'note': 'just wind'},
        {'timestamp': 505, 'has_bee': False, 'note': 'just wind'},
        {'timestamp': 515, 'has_bee': False, 'note': 'just wind'},
        {'timestamp': 525, 'has_bee': False, 'note': 'just wind'},
        {'timestamp': 535, 'has_bee': False, 'note': 'just wind'},
    ]
    
    bee_frames = len([a for a in manual_annotations if a['has_bee']])
    no_bee_frames = len([a for a in manual_annotations if not a['has_bee']])
    
    print(f"📊 MANUAL ANNOTATION SUMMARY:")
    print(f"   Total annotations: {len(manual_annotations)}")
    print(f"   Frames with bees: {bee_frames}")
    print(f"   Frames without bees: {no_bee_frames}")
    print(f"   Peak activity: 315-390s (multiple bees simultaneously)")
    print(f"   Unique bees observed: 11")
    
    return manual_annotations


def main():
    """Main function - create dataset from manual annotations"""
    print(f"🌸 VENTURA MILK VETCH OBJECT DETECTION DATASET")
    print(f"{'='*60}")
    
    # Correct video path for Ventura Milk Vetch
    video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milkvetch/week 5/day 1/site 1/afternoon/P1000446.MP4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found at: {video_path}")
        print("Trying alternative paths...")
        
        # Try alternative paths
        possible_paths = [
            "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milkvetch/week 5/day 1/site 1/afternoon/P1000446.MP4",
            "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milk_vetch/week 5/day 1/site 1/afternoon/P1000446.MP4",
            "/Volumes/Expansion/summer2025_ncos_kb_collections/week 5/day 1/site 1/afternoon/P1000446.MP4",
        ]
        
        video_found = False
        for alt_path in possible_paths:
            if os.path.exists(alt_path):
                video_path = alt_path
                print(f"✅ Found video at: {video_path}")
                video_found = True
                break
        
        if not video_found:
            print("❌ Could not find the correct P1000446.MP4")
            return False
    else:
        print(f"✅ Found video: {video_path}")
    
    print("\n🎬 Creating dataset from your manual annotations...")
    print("📝 Using your manual observations of 11 different bombus bees")
    
    # Get your manual annotations
    manual_annotations = create_manual_annotations()
    
    # Create dataset
    dataset_creator = VenturaMilkVetchDataset()
    extracted_frames, config_path = dataset_creator.prepare_ventura_dataset(manual_annotations)
    
    if extracted_frames:
        print(f"\n✅ DATASET CREATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Dataset location: {dataset_creator.output_dir}")
        print(f"Images: {len(extracted_frames)} frames extracted")
        print(f"Bee frames: {len([f for f in extracted_frames if f['has_bee']])}")
        print(f"Plant type: Ventura Milk Vetch (excellent bee visibility)")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Annotate bounding boxes around bees:")
        print(f"   python3 simple_annotation_script.py")
        print(f"2. Train object detection model:")
        print(f"   python3 object_detection_trainer.py --train")
        print(f"3. Test on videos for accurate bee counting!")
        
        return True
    else:
        print("❌ Dataset creation failed")
        return False

if __name__ == "__main__":
    main()