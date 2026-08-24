#!/usr/bin/env python3
"""
Train the YOLOv8 bee detector (pipeline stage 4 of 6).

Consumes the split dataset written by ``extract_frames.py`` and refined with
``annotate.py``, and writes an Ultralytics run directory (weights plus a
per-epoch ``results.csv``) under ``<data-dir>/models/``.

This is the script that produced the published ``ventura_bee_detection_v2``
run: yolov8s, 150 epochs, batch 8, imgsz 640, lr0 0.01, rect=False.

Usage:
    python src/train_model.py --data-dir data --weights yolov8s.pt --epochs 150

Next stage: ``evaluate.py``.
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
    def __init__(self, data_dir="data", cache_dir="cache"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "yolo_dataset"
        self.models_dir = self.data_dir / "models"
        self.results_dir = self.data_dir / "results"
        self.cache_dir = Path(cache_dir)
        
        for dir_path in [self.train_dir, self.models_dir, self.results_dir, self.cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"🚀 YOLOV8 VENTURA BEE TRAINER")
        print(f"Data directory: {self.data_dir}")
        print(f"Training dataset: {self.train_dir}")
        print(f"Models output: {self.models_dir}")
        print(f"Results output: {self.results_dir}")
        print(f"Cache directory: {self.cache_dir}")
    
    def prepare_yolo_dataset(self, train_split=0.7, val_split=0.2, test_split=0.1):
        print(f"\n📂 PREPARING YOLO DATASET")
        print(f"Train: {train_split*100}% | Val: {val_split*100}% | Test: {test_split*100}%")
        
        images_dir = self.data_dir / "extracted_frames"
        label_dirs = [self.data_dir / "annotations_P1000474", self.data_dir / "annotations"]  # Prioritize P1000474 annotations
        
        image_files = list(images_dir.glob("*.jpg"))
        print(f"Found {len(image_files)} annotated frames")
        
        if not image_files:
            raise FileNotFoundError("No image files found in data/extracted_frames")
        
        expected_prefixes = ["P1000471_ventura_t", "P1000474_t"]
        valid_image_files = [f for f in image_files if any(f.name.startswith(prefix) for prefix in expected_prefixes)]
        if len(valid_image_files) != len(image_files):
            print(f"⚠️ Warning: Found {len(image_files) - len(valid_image_files)} unexpected files in {images_dir}")
            for f in image_files:
                if f not in valid_image_files:
                    print(f"Unexpected file: {f.name}")
        image_files = valid_image_files
        print(f"Using {len(image_files)} valid frames (P1000471 + P1000474)")
        
        if len(image_files) < 165:
            print(f"⚠️ Expected ~165 frames, but found {len(image_files)}. Check for missing or corrupt files.")
        
        random.shuffle(image_files)
        
        n_total = len(image_files)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        for split in ['train', 'val', 'test']:
            (self.train_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.train_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        skipped_files = []
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            print(f"Copying {len(files)} files to {split_name}...")
            
            for img_file in files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        print(f"⚠️ Corrupt image: {img_file.name}")
                        skipped_files.append(img_file.name)
                        continue
                except Exception as e:
                    print(f"⚠️ Failed to read image {img_file.name}: {e}")
                    skipped_files.append(img_file.name)
                    continue
                
                dst_img = self.train_dir / split_name / 'images' / img_file.name
                try:
                    shutil.copy2(img_file, dst_img)
                except Exception as e:
                    print(f"⚠️ Failed to copy image {img_file.name}: {e}")
                    skipped_files.append(img_file.name)
                    continue
                
                label_file = None
                for label_dir in label_dirs:
                    possible_label = label_dir / (img_file.stem + '.txt')
                    if possible_label.exists():
                        label_file = possible_label
                        break
                if label_file:
                    dst_label = self.train_dir / split_name / 'labels' / label_file.name
                    try:
                        shutil.copy2(label_file, dst_label)
                    except Exception as e:
                        print(f"⚠️ Failed to copy label {label_file.name}: {e}")
                        skipped_files.append(label_file.name)
                        continue
                else:
                    print(f"⚠️ No label file found for {img_file.name}")
                    skipped_files.append(img_file.name)
        
        if skipped_files:
            print(f"⚠️ Skipped {len(skipped_files)} files due to errors: {skipped_files}")
        
        dataset_config = {
            'path': str(self.train_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['bee'],
            'cache': str(self.cache_dir / 'labels.cache')
        }
        
        config_path = self.train_dir / 'dataset.yaml'
        try:
            with open(config_path, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False)
        except Exception as e:
            print(f"⚠️ Failed to create dataset.yaml: {e}")
            raise
        
        print(f"✅ Dataset prepared: {config_path}")
        return config_path
    
    def train_model(self, model_path='yolov8s.pt', epochs=200, img_size=640, batch_size=4):
        print(f"\n🏋️ TRAINING YOLOV8S MODEL")
        print(f"Model path: {model_path}")
        print(f"Epochs: {epochs}, Image size: {img_size}, Batch size: {batch_size}")
        
        config_path = self.prepare_yolo_dataset()
        
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"❌ Failed to load model {model_path}: {e}")
            print("Falling back to default yolov8s.pt")
            model = YOLO('yolov8s.pt')
        
        try:
            results = model.train(
                data=str(config_path),
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                name='ventura_bee_detection_v23',  # Updated to v23
                project=str(self.models_dir),
                save=True,
                plots=True,
                device='cpu',
                patience=100,
                save_period=20,
                lr0=0.005,
                lrf=0.1,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=10.0,
                translate=0.1,
                scale=0.5,
                shear=5.0,
                perspective=0.0001,
                flipud=0.5,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.3,
                copy_paste=0.0
            )
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
        
        best_model_path = self.models_dir / 'ventura_bee_detection_v23' / 'weights' / 'best.pt'
        
        print(f"✅ Training complete!")
        print(f"Best model: {best_model_path}")
        
        return best_model_path, results
    
    def validate_model(self, model_path):
        print(f"\n🔍 VALIDATING MODEL")
        
        model = YOLO(model_path)
        
        try:
            val_results = model.val(
                data=str(self.train_dir / 'dataset.yaml'),
                save_json=True,
                save_hybrid=False
            )
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return None
        
        print(f"📊 VALIDATION RESULTS:")
        print(f" mAP50: {val_results.box.map50:.3f}")
        print(f" mAP50-95: {val_results.box.map:.3f}")
        print(f" Precision: {val_results.box.mp:.3f}")
        print(f" Recall: {val_results.box.mr:.3f}")
        
        return val_results
    
    def test_on_sample_images(self, model_path, num_samples=10):
        print(f"\n🧪 TESTING ON SAMPLE IMAGES")
        
        model = YOLO(model_path)
        test_images_dir = self.train_dir / 'test' / 'images'
        test_images = list(test_images_dir.glob("*.jpg"))[:num_samples]
        
        results_images = []
        
        for img_path in test_images:
            try:
                results = model(str(img_path))
                annotated_img = results[0].plot()
                output_path = self.results_dir / f"test_{img_path.stem}.jpg"
                cv2.imwrite(str(output_path), annotated_img)
                results_images.append(output_path)
                print(f" Processed: {img_path.name} -> {output_path.name}")
            except Exception as e:
                print(f"⚠️ Failed to process {img_path.name}: {e}")
        
        print(f"✅ Test images saved to: {self.results_dir}")
        return results_images
    
    def create_training_summary(self, model_path, val_results):
        print(f"\n📈 CREATING TRAINING SUMMARY")
        
        training_dir = self.models_dir / 'ventura_bee_detection_v23'  # Updated to v23
        results_csv = training_dir / 'results.csv'
        
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('YOLOv8 Ventura Bee Detection Training Summary', fontsize=16)
            
            if 'train/box_loss' in df.columns:
                ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
                ax1.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Box Loss')
                ax1.set_title('Box Loss Over Time')
                ax1.legend()
                ax1.grid(True)
            
            if 'metrics/mAP50(B)' in df.columns:
                ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
                ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('mAP')
                ax2.set_title('Mean Average Precision')
                ax2.legend()
                ax2.grid(True)
            
            if 'metrics/precision(B)' in df.columns:
                ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
                ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Score')
                ax3.set_title('Precision & Recall')
                ax3.legend()
                ax3.grid(True)
            
            final_metrics = f"""
            Final Training Results:
            
            mAP50: {val_results.box.map50:.3f if val_results else 'N/A'}
            mAP50-95: {val_results.box.map:.3f if val_results else 'N/A'}
            Precision: {val_results.box.mp:.3f if val_results else 'N/A'}
            Recall: {val_results.box.mr:.3f if val_results else 'N/A'}
            
            Model: {model_path.name}
            Training Epochs: {len(df)}
            Dataset: Ventura Milkvetch (P1000471 + P1000474)
            Classes: 1 (bee)
            """
            
            ax4.text(0.1, 0.9, final_metrics, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
            ax4.set_title('Training Summary')
            ax4.axis('off')
            
            plt.tight_layout()
            
            summary_path = self.results_dir / 'training_summary.jpg'
            try:
                plt.savefig(summary_path, dpi=150, bbox_inches='tight')
                plt.show()
            except Exception as e:
                print(f"⚠️ Failed to save training summary: {e}")
            
            print(f"✅ Training summary saved: {summary_path}")
            return summary_path
        else:
            print(f"⚠️ Results CSV not found: {results_csv}")
            return None
    
    def run_full_training_pipeline(self, model_path='yolov8s.pt', epochs=200):
        print(f"\n🚀 STARTING TRAINING PIPELINE")
        print(f"{'='*60}")
        print(f"Model: YOLOv8s (fine-tuning from {model_path})")
        print(f"Epochs: {epochs}")
        print(f"Focus: Training on combined P1000471 and P1000474 datasets")
        
        model_path, train_results = self.train_model(
            model_path=model_path,
            epochs=epochs,
            img_size=640,
            batch_size=4
        )
        
        val_results = self.validate_model(model_path)
        
        test_images = self.test_on_sample_images(model_path, num_samples=10)
        
        summary_path = self.create_training_summary(model_path, val_results)
        
        print(f"\n✅ TRAINING PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"🎯 Best model: {model_path}")
        print(f"📊 Validation mAP50: {val_results.box.map50:.3f if val_results else 'N/A'}")
        print(f"📊 Precision: {val_results.box.mp:.3f if val_results else 'N/A'}")
        print(f"📊 Recall: {val_results.box.mr:.3f if val_results else 'N/A'}")
        print(f"📁 Results: {self.results_dir}")
        print(f"🖼️ Test images: {len(test_images)} samples")
        
        return {
            'model_path': model_path,
            'val_results': val_results,
            'test_images': test_images,
            'summary_path': summary_path
        }

def main(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed. Install with:")
        print("pip install -U ultralytics")
        return
    
    data_dir = args.data_dir
    if not os.path.exists(f"{data_dir}/extracted_frames"):
        print(f"❌ No extracted frames found in {data_dir}/extracted_frames")
        return
    if not os.path.exists(f"{data_dir}/annotations"):
        print(f"❌ No annotations found in {data_dir}/annotations")
        return
    
    model_path = args.weights
    if not os.path.exists(model_path):
        print(f"❌ Pre-trained model not found at {model_path}. Using default yolov8s.pt")
        model_path = "yolov8s.pt"
    
    import shutil
    disk_usage = shutil.disk_usage(data_dir)
    free_space_mb = disk_usage.free / (1024 * 1024)
    if free_space_mb < 5000:
        print(f"⚠️ Low disk space: {free_space_mb:.2f} MB free. Recommend at least 5000 MB.")
    
    trainer = VenturaYOLOTrainer(data_dir, cache_dir=args.cache_dir)
    
    try:
        results = trainer.run_full_training_pipeline(
            model_path=model_path,
            epochs=args.epochs
        )
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return
    
    print(f"\n🎉 Training complete! Model ready for video testing.")
    print(f"Next step: Run 4_test_videos.py to test on multiple videos")

def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="Train a YOLOv8 bee detector on the Ventura milk-vetch frame set.")
    ap.add_argument("--data-dir", default="data",
                    help="Directory holding yolo_dataset/, models/, results/ (default: ./data)")
    ap.add_argument("--cache-dir", default="cache",
                    help="Scratch directory for the label cache (default: ./cache)")
    ap.add_argument("--weights", default="yolov8s.pt",
                    help="Starting weights: an ultralytics model name or a path to a .pt checkpoint")
    ap.add_argument("--epochs", type=int, default=150,
                    help="Training epochs (the published v2 run used 150)")
    return ap.parse_args()


if __name__ == "__main__":
    main(_parse_args())