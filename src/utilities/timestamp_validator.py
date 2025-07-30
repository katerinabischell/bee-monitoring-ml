#!/usr/bin/env python3
"""
Validate AI predictions against manual annotations at specific timestamps
"""

import cv2
import json
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image

try:
    from correct_video_analyzer import BombusClassifier
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

class TimestampValidator:
    """
    Validate AI predictions against manual annotations
    """
    
    def __init__(self, model_path="best_bombus_model.pth"):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the model"""
        if not MODEL_AVAILABLE:
            return
        
        try:
            self.model = BombusClassifier(num_classes=2)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("✅ Model loaded for timestamp validation")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def test_specific_timestamps(self, video_path, timestamps):
        """
        Test model predictions at specific timestamps
        
        Args:
            video_path: Path to video
            timestamps: List of timestamps in seconds to test
        """
        if not self.model:
            print("❌ Model not available")
            return
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = []
        
        print(f"\n🔍 Testing AI predictions at specific timestamps")
        print(f"Video: {Path(video_path).name}")
        print("-" * 60)
        
        for timestamp in timestamps:
            # Jump to specific frame
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Could not read frame at {timestamp}s")
                continue
            
            # Get AI prediction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            image_tensor = self.transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                bombus_prob = probabilities[0][1].item()
                has_bombus = bombus_prob > 0.5
            
            # Convert timestamp to MM:SS format
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes}:{seconds:02d}"
            
            result = {
                'timestamp': timestamp,
                'time_formatted': time_str,
                'ai_prediction': has_bombus,
                'ai_confidence': bombus_prob,
                'frame_number': frame_number
            }
            
            results.append(result)
            
            # Print result
            prediction_text = "🐝 BEE DETECTED" if has_bombus else "❌ NO BEE"
            print(f"{time_str:>6s} | {prediction_text:>15s} | Confidence: {bombus_prob:.3f}")
        
        cap.release()
        return results
    
    def validate_against_manual_annotations(self, video_path):
        """
        Validate against your manual annotations for P1000421.MP4
        """
        # Your manual annotations converted to timestamps
        manual_annotations = [
            # Bee activity period (should detect bees)
            {'time': 102, 'expected': True, 'note': 'Bombus present on flower (middle)'},
            {'time': 105, 'expected': True, 'note': 'Bombus flies up'},
            {'time': 120, 'expected': True, 'note': 'Bombus on plant'},
            {'time': 130, 'expected': True, 'note': 'Second bombus lands (left side)'},
            {'time': 148, 'expected': True, 'note': 'Second bombus visible (left side)'},
            {'time': 170, 'expected': True, 'note': 'Second bombus lands with orange pollen'},
            {'time': 182, 'expected': True, 'note': 'Second bombus visible (left side)'},
            {'time': 202, 'expected': True, 'note': 'Second bombus with bright orange pollen'},
            
            # No bee periods (should NOT detect bees)
            {'time': 30, 'expected': False, 'note': 'Wind blowing, flowers moving'},
            {'time': 80, 'expected': False, 'note': 'Wind blowing, flowers moving'},
            {'time': 224, 'expected': False, 'note': 'Wind blowing, flowers moving'},
            {'time': 290, 'expected': False, 'note': 'Wind blowing, flowers moving'},
            {'time': 365, 'expected': False, 'note': 'Wind blowing, flowers moving'},
            {'time': 457, 'expected': False, 'note': 'Wind blowing, flowers moving'},
            {'time': 500, 'expected': False, 'note': 'Sound of airplane, wind blowing'},
        ]
        
        print(f"\n{'='*70}")
        print(f"VALIDATION AGAINST MANUAL ANNOTATIONS")
        print(f"{'='*70}")
        
        # Test all annotated timestamps
        timestamps = [ann['time'] for ann in manual_annotations]
        ai_results = self.test_specific_timestamps(video_path, timestamps)
        
        # Compare results
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"{'Time':>6s} | {'Expected':>10s} | {'AI Result':>10s} | {'Match':>8s} | {'Confidence':>10s} | Notes")
        print("-" * 85)
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, ann in enumerate(manual_annotations):
            if i < len(ai_results):
                ai_result = ai_results[i]
                expected = ann['expected']
                ai_pred = ai_result['ai_prediction']
                confidence = ai_result['ai_confidence']
                
                match = "✅" if expected == ai_pred else "❌"
                if expected == ai_pred:
                    correct_predictions += 1
                total_predictions += 1
                
                expected_text = "BEE" if expected else "NO BEE"
                ai_text = "BEE" if ai_pred else "NO BEE"
                time_str = ai_result['time_formatted']
                
                print(f"{time_str:>6s} | {expected_text:>10s} | {ai_text:>10s} | {match:>8s} | {confidence:>10.3f} | {ann['note'][:30]}")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"\n📈 VALIDATION SUMMARY:")
        print(f"   Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Model performance: {'Good' if accuracy > 0.8 else 'Needs improvement'}")
        
        return ai_results, manual_annotations, accuracy


def main():
    """Run timestamp validation"""
    video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/birds_beak/week_5/day_2/site_4/afternoon/P1000421.MP4"
    
    validator = TimestampValidator()
    
    if not validator.model:
        print("❌ Cannot run validation without model")
        return
    
    # Run validation against manual annotations
    ai_results, manual_annotations, accuracy = validator.validate_against_manual_annotations(video_path)
    
    print(f"\n💡 INSIGHTS:")
    if accuracy < 0.7:
        print("   - Model needs retraining with better annotations")
        print("   - Consider what visual features the model learned")
        print("   - May need more diverse training data")
    else:
        print("   - Model performs reasonably well")
        print("   - Fine-tuning could improve edge cases")
    
    print(f"\n✅ Validation complete!")
    print(f"This analysis shows exactly where your model succeeds and fails")

if __name__ == "__main__":
    main()