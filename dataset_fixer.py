#!/usr/bin/env python3
"""
Fix the dataset structure to include validation split for YOLO training
"""

import os
import shutil
import random
from pathlib import Path
import yaml

def fix_dataset_structure():
    """
    Split the training data into train/val splits and fix the dataset config
    """
    
    dataset_dir = Path("ventura_object_detection_dataset")
    
    print("🔧 FIXING DATASET STRUCTURE FOR YOLO")
    print("="*50)
    
    # Check current structure
    train_images = list((dataset_dir / "images" / "train").glob("*.jpg"))
    train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
    
    print(f"Current training data:")
    print(f"   Images: {len(train_images)}")
    print(f"   Labels: {len(train_labels)}")
    
    if len(train_images) == 0:
        print("❌ No training images found!")
        return False
    
    # Create val directories
    val_img_dir = dataset_dir / "images" / "val"
    val_label_dir = dataset_dir / "labels" / "val"
    val_img_dir.mkdir(exist_ok=True)
    val_label_dir.mkdir(exist_ok=True)
    
    # Split data: 80% train, 20% val
    random.seed(42)  # For reproducible splits
    all_images = sorted(train_images)
    random.shuffle(all_images)
    
    split_idx = int(0.8 * len(all_images))
    train_split = all_images[:split_idx]
    val_split = all_images[split_idx:]
    
    print(f"\nSplitting data:")
    print(f"   Train: {len(train_split)} images")
    print(f"   Val: {len(val_split)} images")
    
    # Move validation images and labels
    moved_val = 0
    for img_path in val_split:
        # Move image
        val_img_path = val_img_dir / img_path.name
        shutil.move(str(img_path), str(val_img_path))
        
        # Move corresponding label
        label_name = img_path.stem + ".txt"
        label_path = dataset_dir / "labels" / "train" / label_name
        val_label_path = val_label_dir / label_name
        
        if label_path.exists():
            shutil.move(str(label_path), str(val_label_path))
            moved_val += 1
        else:
            # Create empty label file for images without annotations
            val_label_path.touch()
    
    print(f"✅ Moved {moved_val} image-label pairs to validation set")
    
    # Update dataset.yaml
    dataset_config = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',  # Use val as test for now
        'nc': 1,
        'names': ['bombus']
    }
    
    config_path = dataset_dir / "dataset.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Updated dataset.yaml")
    
    # Verify final structure
    print(f"\n📊 FINAL DATASET STRUCTURE:")
    train_images_final = len(list((dataset_dir / "images" / "train").glob("*.jpg")))
    val_images_final = len(list((dataset_dir / "images" / "val").glob("*.jpg")))
    train_labels_final = len(list((dataset_dir / "labels" / "train").glob("*.txt")))
    val_labels_final = len(list((dataset_dir / "labels" / "val").glob("*.txt")))
    
    print(f"   Train: {train_images_final} images, {train_labels_final} labels")
    print(f"   Val: {val_images_final} images, {val_labels_final} labels")
    print(f"   Total: {train_images_final + val_images_final} images")
    
    return True

def verify_annotations():
    """
    Check that annotations are properly formatted
    """
    dataset_dir = Path("ventura_object_detection_dataset")
    
    print(f"\n🔍 VERIFYING ANNOTATIONS:")
    
    total_annotations = 0
    files_with_annotations = 0
    
    for split in ['train', 'val']:
        label_dir = dataset_dir / "labels" / split
        
        if not label_dir.exists():
            continue
        
        label_files = list(label_dir.glob("*.txt"))
        split_annotations = 0
        split_files_with_annotations = 0
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                
            if len(lines) > 0:
                split_files_with_annotations += 1
                split_annotations += len(lines)
        
        print(f"   {split.capitalize()}: {split_annotations} annotations in {split_files_with_annotations}/{len(label_files)} files")
        total_annotations += split_annotations
        files_with_annotations += split_files_with_annotations
    
    print(f"   Total: {total_annotations} bee annotations across {files_with_annotations} files")
    
    return total_annotations > 0

def main():
    """Fix the dataset and verify it's ready for training"""
    
    if not fix_dataset_structure():
        return False
    
    if not verify_annotations():
        print("❌ No annotations found! Check your annotation files.")
        return False
    
    print(f"\n✅ DATASET READY FOR TRAINING!")
    print(f"{'='*50}")
    print(f"Next step: python3 object_detection_trainer.py --train")
    
    return True

if __name__ == "__main__":
    main()