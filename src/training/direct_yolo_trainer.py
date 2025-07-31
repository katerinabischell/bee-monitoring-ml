#!/usr/bin/env python3
"""
Direct YOLO training script - uses existing Ventura dataset without recreating it
"""

import os
from pathlib import Path

def train_ventura_model():
    """
    Train YOLO model directly on the existing Ventura dataset
    """
    
    print("🚀 DIRECT YOLO TRAINING")
    print("="*50)
    
    # Check that dataset exists
    dataset_dir = Path("ventura_object_detection_dataset")
    config_path = dataset_dir / "dataset.yaml"
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    if not config_path.exists():
        print(f"❌ Dataset config not found: {config_path}")
        return False
    
    # Check dataset structure
    train_imgs = list((dataset_dir / "images" / "train").glob("*.jpg"))
    val_imgs = list((dataset_dir / "images" / "val").glob("*.jpg"))
    train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
    val_labels = list((dataset_dir / "labels" / "val").glob("*.txt"))
    
    print(f"📊 Dataset verification:")
    print(f"   Train: {len(train_imgs)} images, {len(train_labels)} labels")
    print(f"   Val: {len(val_imgs)} images, {len(val_labels)} labels")
    
    if len(train_imgs) == 0:
        print("❌ No training images found!")
        return False
    
    if len(val_imgs) == 0:
        print("❌ No validation images found!")
        return False
    
    # Count annotations
    total_annotations = 0
    for label_file in train_labels + val_labels:
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            total_annotations += len(lines)
    
    print(f"   Total bee annotations: {total_annotations}")
    
    if total_annotations == 0:
        print("❌ No annotations found!")
        return False
    
    # Install ultralytics if needed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("📦 Installing ultralytics...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO
    
    print(f"\n🚀 Starting YOLO training...")
    print(f"   Using dataset: {config_path}")
    print(f"   Model: YOLOv8n (nano)")
    print(f"   Epochs: 50")
    print(f"   Device: CPU")
    
    try:
        # Load pretrained model
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=str(config_path),
            epochs=50,
            imgsz=640,
            project="models/object_detection",
            name="ventura_bombus_yolo",
            device='cpu',
            patience=20,
            save_period=10,
            verbose=True
        )
        
        # Find the trained model
        model_dir = Path("models/object_detection/ventura_bombus_yolo")
        model_path = model_dir / "weights" / "best.pt"
        
        if model_path.exists():
            print(f"\n✅ TRAINING COMPLETE!")
            print(f"📁 Model saved to: {model_path}")
            
            # Test the model
            test_trained_model(str(model_path))
            
            return True
        else:
            print(f"❌ Model file not found at: {model_path}")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def test_trained_model(model_path):
    """
    Test the trained model on the Ventura video
    """
    
    print(f"\n🧪 TESTING TRAINED MODEL")
    print("="*40)
    
    try:
        from ultralytics import YOLO
        
        # Load trained model
        model = YOLO(model_path)
        
        # Test video path
        test_video = "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milkvetch/week 5/day 1/site 1/afternoon/P1000446.MP4"
        
        if not os.path.exists(test_video):
            print(f"⚠️ Test video not found: {test_video}")
            return
        
        print(f"🎬 Testing on: P1000446.MP4")
        
        # Run prediction on a few frames
        results = model.predict(
            source=test_video,
            conf=0.5,
            save=False,
            show=False,
            verbose=False
        )
        
        # Analyze results
        total_detections = 0
        frames_with_bees = 0
        max_bees_in_frame = 0
        
        frame_count = 0
        for result in results:
            frame_count += 1
            
            if result.boxes is not None:
                bee_count = len(result.boxes)
                if bee_count > 0:
                    frames_with_bees += 1
                    total_detections += bee_count
                    max_bees_in_frame = max(max_bees_in_frame, bee_count)
            
            # Stop after 300 frames (10 seconds) for quick test
            if frame_count >= 300:
                break
        
        detection_rate = frames_with_bees / frame_count if frame_count > 0 else 0
        
        print(f"\n📊 TEST RESULTS (first 10 seconds):")
        print(f"   Frames analyzed: {frame_count}")
        print(f"   Total bee detections: {total_detections}")
        print(f"   Frames with bees: {frames_with_bees}")
        print(f"   Detection rate: {detection_rate:.1%}")
        print(f"   Max bees in single frame: {max_bees_in_frame}")
        
        if total_detections > 0:
            print(f"🎉 Model successfully detecting bees!")
        else:
            print(f"⚠️ No bees detected - may need more training or different thresholds")
            
    except Exception as e:
        print(f"❌ Testing error: {e}")

def main():
    """
    Main function - direct training without dataset recreation
    """
    
    print("🐝 DIRECT VENTURA BOMBUS YOLO TRAINING")
    print("="*60)
    print("Using existing annotated Ventura Milk Vetch dataset")
    print("="*60)
    
    success = train_ventura_model()
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"Your YOLO object detection model is ready!")
        print(f"It can now count individual bees and locate them in videos.")
    else:
        print(f"\n❌ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main()