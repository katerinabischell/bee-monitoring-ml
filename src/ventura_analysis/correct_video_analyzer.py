"""
Video Analyzer - Using the CORRECT architecture that matches your trained model
Author: Katerina Bischel
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from pathlib import Path

class BombusClassifier(nn.Module):
    """CNN model for bombus detection - CORRECT architecture matching your saved model"""
    
    def __init__(self, num_classes=2):
        super(BombusClassifier, self).__init__()
        
        # This matches the actual architecture that was saved
        # Your model used ResNet18 backbone, not custom CNN
        self.backbone = models.resnet18(pretrained=False)  # Don't need pretrained weights for loading
        
        # Replace final layer for classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def test_model_on_video(video_path, model_path='best_bombus_model.pth'):
    """Test your trained model on a video"""
    
    print(f"🎬 Testing model on: {Path(video_path).name}")
    
    # Load model with CORRECT architecture
    model = BombusClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Image preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
    
    # Test on frames throughout the video
    num_test_frames = min(30, total_frames // 10)  # Test 30 frames or every 10th frame
    test_frames = np.linspace(0, total_frames-1, num_test_frames, dtype=int)
    
    bombus_detections = 0
    high_confidence_detections = 0
    all_confidences = []
    detection_times = []
    
    print(f"🔍 Analyzing {num_test_frames} frames...")
    
    for i, frame_idx in enumerate(test_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_image).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            is_bombus = predicted.item() == 1
            conf_score = confidence.item()
            all_confidences.append(conf_score if is_bombus else 0)
            
            if is_bombus:
                bombus_detections += 1
                timestamp = frame_idx / fps if fps > 0 else 0
                detection_times.append(timestamp)
                
                if conf_score > 0.8:  # High confidence threshold
                    high_confidence_detections += 1
                    print(f"🐝 HIGH CONFIDENCE BOMBUS at {timestamp:.1f}s (confidence: {conf_score:.3f})")
                else:
                    print(f"🐝 Bombus detected at {timestamp:.1f}s (confidence: {conf_score:.3f})")
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{num_test_frames} frames...")
    
    cap.release()
    
    # Analysis summary
    detection_rate = bombus_detections / num_test_frames
    avg_confidence = np.mean([c for c in all_confidences if c > 0]) if bombus_detections > 0 else 0
    
    print(f"\n" + "="*50)
    print(f"📈 ANALYSIS RESULTS")
    print("="*50)
    print(f"Video: {Path(video_path).name}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Frames analyzed: {num_test_frames}")
    print(f"Bombus detections: {bombus_detections}")
    print(f"High confidence detections: {high_confidence_detections}")
    print(f"Detection rate: {detection_rate:.1%}")
    
    if bombus_detections > 0:
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Detection times: {[f'{t:.1f}s' for t in detection_times]}")
        print(f"\n🐝 BOMBUS ACTIVITY DETECTED!")
        print(f"   This video shows pollinator visits to your plants.")
        
        if high_confidence_detections > 0:
            print(f"   {high_confidence_detections} high-confidence detections suggest")
            print(f"   clear, unambiguous bombus presence.")
    else:
        print(f"\n📹 No bombus activity detected in analyzed frames.")
        print(f"   This could mean:")
        print(f"   • No bombus visited during this recording")
        print(f"   • Bombus activity occurred between analyzed frames")
        print(f"   • Video quality/lighting affected detection")
    
    return {
        'detection_rate': detection_rate,
        'bombus_detections': bombus_detections,
        'high_confidence': high_confidence_detections,
        'avg_confidence': avg_confidence,
        'detection_times': detection_times
    }

def quick_test():
    """Quick test to verify model loading"""
    print("🧪 Quick model test...")
    
    try:
        model = BombusClassifier(num_classes=2)
        model.load_state_dict(torch.load('best_bombus_model.pth', map_location='cpu'))
        model.eval()
        print("✅ Model architecture matches saved weights!")
        print("🚀 Ready to analyze videos!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Quick test first
    if not quick_test():
        sys.exit(1)
    
    # Check arguments
    if len(sys.argv) != 2:
        print("\nUsage:")
        print("  python3 correct_video_analyzer.py /path/to/video.mp4")
        print("\nExample:")
        print('  python3 correct_video_analyzer.py "/Volumes/Expansion/summer2025_ncos_kb_collections/birds_beak/week_2/day_1/site_1/morning/P1000372.MP4"')
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    # Analyze the video
    results = test_model_on_video(video_path)