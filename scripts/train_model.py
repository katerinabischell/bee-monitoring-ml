#!/usr/bin/env python3
"""
YOLOv8 Training Pipeline for Ventura Milkvetch Bee Detection
"""

import os
import shutil
import yaml
from pathlib import Path
import random
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

class VenturaYOLOTrainer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "yolo_dataset"
        self.models_dir = self.data_dir / "models"
        self.results_dir = self.data_dir / "results"
        
        # Create directories
        for dir_path in [self.train_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"🚀 YOLOV8 VENTURA BEE TRAINER")
        print(f"Data directory: {self.data_dir}")
        print(f"Training dataset: {self.train_dir}")
        print(f"Models output: {self.models_dir}")
        print(f"Results output: {self.results_dir}")
    
    def prepare_yolo_dataset(self, train_split=0.7, val_split=0.2, test_split=0.1):
        """Prepare YOLO dataset structure from annotations"""
        print(f"\n📂 PREPARING YOLO DATASET")
        print(f"Train: {train_split*100}% | Val: {val_split*100}% | Test: {test_split*100}%")
        
        # Source directories
        images_dir = self.data_dir / "extracted_frames"
        labels_dir = self.data_dir / "annotations"
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg"))
        print(f"Found {len(image_files)} annotated frames")
        
        # Shuffle for random split
        random.shuffle(image_files)
        
        # Calculate split indices
        n_total = len(image_files)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (self.train_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.train_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy files to appropriate splits
        splits = {
            'train': train_files,
            'val': val_files, 
            'test': test_files
        }
        
        for split_name, files in splits.items():
            print(f"Copying {len(files)} files to {split_name}...")
            
            for img_file in files:
                # Copy image
                dst_img = self.train_dir / split_name / 'images' / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy corresponding label
                label_file = labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    dst_label = self.train_dir / split_name / 'labels' / label_file.name
                    shutil.copy2(label_file, dst_label)
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(self.train_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['bee']
        }
        
        config_path = self.train_dir / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Dataset prepared: {config_path}")
        return config_path
    
    def train_model(self, model_size='n', epochs=150, img_size=640, batch_size=16):
        """Train YOLOv8 model"""
        print(f"\n🏋️ TRAINING YOLOV8{model_size.upper()} MODEL")
        print(f"Epochs: {epochs}, Image size: {img_size}, Batch size: {batch_size}")
        
        # Prepare dataset
        config_path = self.prepare_yolo_dataset()
        
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')  # Load pretrained model
        
        # Train with simplified but effective hyperparameters
        results = model.train(
            data=str(config_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='ventura_bee_detection_v2',
            project=str(self.models_dir),
            save=True,
            plots=True,
            device='cpu',  # Change to 'cuda' if you have GPU
            patience=30,  # Increased patience for longer training
            save_period=20,  # Save checkpoint every 20 epochs
            # Core hyperparameters (using valid names)
            lr0=0.01,      # Learning rate
            lrf=0.1,       # Final learning rate (fraction of lr0)
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,       # Box loss gain
            cls=0.5,       # Class loss gain
            dfl=1.5,       # Distribution focal loss gain
            # Data augmentation
            hsv_h=0.015,   # Image HSV-Hue augmentation
            hsv_s=0.7,     # Image HSV-Saturation augmentation  
            hsv_v=0.4,     # Image HSV-Value augmentation
            degrees=0.0,   # Image rotation (+/- deg)
            translate=0.1, # Image translation (+/- fraction)
            scale=0.5,     # Image scale (+/- gain)
            shear=0.0,     # Image shear (+/- deg)
            perspective=0.0, # Image perspective (+/- fraction)
            flipud=0.0,    # Image flip up-down (probability)
            fliplr=0.5,    # Image flip left-right (probability)
            mosaic=1.0,    # Image mosaic (probability)
            mixup=0.0,     # Image mixup (probability)
            copy_paste=0.0 # Segment copy-paste (probability)
        )
        
        # Get best model path
        best_model_path = self.models_dir / 'ventura_bee_detection_v2' / 'weights' / 'best.pt'
        
        print(f"✅ Training complete!")
        print(f"Best model: {best_model_path}")
        
        return best_model_path, results
    
    def validate_model(self, model_path):
        """Validate trained model"""
        print(f"\n🔍 VALIDATING MODEL")
        
        model = YOLO(model_path)
        
        # Run validation
        val_results = model.val(
            data=str(self.train_dir / 'dataset.yaml'),
            save_json=True,
            save_hybrid=True,
        )
        
        # Print key metrics
        print(f"📊 VALIDATION RESULTS:")
        print(f"   mAP50: {val_results.box.map50:.3f}")
        print(f"   mAP50-95: {val_results.box.map:.3f}")
        print(f"   Precision: {val_results.box.mp:.3f}")
        print(f"   Recall: {val_results.box.mr:.3f}")
        
        return val_results
    
    def test_on_sample_images(self, model_path, num_samples=10):
        """Test model on sample images"""
        print(f"\n🧪 TESTING ON SAMPLE IMAGES")
        
        model = YOLO(model_path)
        test_images_dir = self.train_dir / 'test' / 'images'
        test_images = list(test_images_dir.glob('*.jpg'))[:num_samples]
        
        results_images = []
        
        for img_path in test_images:
            # Run inference
            results = model(str(img_path))
            
            # Plot results
            annotated_img = results[0].plot()
            
            # Save result
            output_path = self.results_dir / f"test_{img_path.stem}.jpg"
            cv2.imwrite(str(output_path), annotated_img)
            results_images.append(output_path)
            
            print(f"   Processed: {img_path.name} -> {output_path.name}")
        
        print(f"✅ Test images saved to: {self.results_dir}")
        return results_images
    
    def create_training_summary(self, model_path, val_results):
        """Create training summary plot"""
        print(f"\n📈 CREATING TRAINING SUMMARY")
        
        # Read training results
        training_dir = self.models_dir / 'ventura_bee_detection'
        results_csv = training_dir / 'results.csv'
        
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()  # Remove extra spaces
            
            # Create summary plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('YOLOv8 Ventura Bee Detection Training Summary', fontsize=16)
            
            # Loss curves
            if 'train/box_loss' in df.columns:
                ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
                ax1.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Box Loss')
                ax1.set_title('Box Loss Over Time')
                ax1.legend()
                ax1.grid(True)
            
            # mAP curves
            if 'metrics/mAP50(B)' in df.columns:
                ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
                ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('mAP')
                ax2.set_title('Mean Average Precision')
                ax2.legend()
                ax2.grid(True)
            
            # Precision/Recall
            if 'metrics/precision(B)' in df.columns:
                ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
                ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Score')
                ax3.set_title('Precision & Recall')
                ax3.legend()
                ax3.grid(True)
            
            # Final metrics summary
            final_metrics = f"""
            Final Training Results:
            
            mAP50: {val_results.box.map50:.3f}
            mAP50-95: {val_results.box.map:.3f}
            Precision: {val_results.box.mp:.3f}
            Recall: {val_results.box.mr:.3f}
            
            Model: {model_path.name}
            Training Epochs: {len(df)}
            Dataset: Ventura Milkvetch
            Classes: 1 (bee)
            """
            
            ax4.text(0.1, 0.9, final_metrics, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
            ax4.set_title('Training Summary')
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save summary
            summary_path = self.results_dir / 'training_summary.jpg'
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Training summary saved: {summary_path}")
            return summary_path
    
    def run_full_training_pipeline(self, model_size='s', epochs=150):
        """Run complete training pipeline"""
        print(f"\n🚀 STARTING IMPROVED TRAINING PIPELINE")
        print(f"{'='*60}")
        print(f"Model size: YOLOv8{model_size} (larger than nano for better accuracy)")
        print(f"Epochs: {epochs} (more training time)")
        print(f"Focus: Improving recall to catch more bees")
        
        # Train model
        model_path, train_results = self.train_model(
            model_size=model_size, 
            epochs=epochs,
            img_size=640,
            batch_size=8  # Smaller batch for stability
        )
        
        # Validate model  
        val_results = self.validate_model(model_path)
        
        # Test on samples
        test_images = self.test_on_sample_images(model_path, num_samples=10)
        
        # Create summary
        summary_path = self.create_training_summary(model_path, val_results)
        
        print(f"\n✅ IMPROVED TRAINING PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"🎯 Best model: {model_path}")
        print(f"📊 Validation mAP50: {val_results.box.map50:.3f}")
        print(f"📊 Precision: {val_results.box.mp:.3f}")
        print(f"📊 Recall: {val_results.box.mr:.3f}")
        print(f"📁 Results: {self.results_dir}")
        print(f"🖼️ Test images: {len(test_images)} samples")
        
        return {
            'model_path': model_path,
            'val_results': val_results,
            'test_images': test_images,
            'summary_path': summary_path
        }


def main():
    """Run YOLOv8 training"""
    # Check requirements
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed. Install with:")
        print("pip install ultralytics")
        return
    
    # Check data exists
    data_dir = "data"
    if not os.path.exists(f"{data_dir}/extracted_frames"):
        print("❌ No extracted frames found. Run 1_extract_frames.py first")
        return
    
    if not os.path.exists(f"{data_dir}/annotations"):
        print("❌ No annotations found. Run 2_annotate_gui.py first")
        return
    
    # Start training
    trainer = VenturaYOLOTrainer(data_dir)
    
    # Run improved pipeline
    results = trainer.run_full_training_pipeline(
        model_size='s',  # Small model (better than nano)
        epochs=150       # More training epochs
    )
    
    print(f"\n🎉 Training complete! Model ready for video testing.")
    print(f"Next step: Run 4_test_videos.py to test on multiple videos")

if __name__ == "__main__":
    main()