"""
Simple Video Test Script - Test trained model quickly
Author: Katerina Bischel
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from pathlib import Path

class BombusClassifier(nn.Module):
    """CNN model for bombus detection - matches your trained model"""
    
    def __init__(self, num_classes=2):
        super(BombusClassifier, self).__init__()
        
        # Custom CNN architecture (matches your trained model)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def test_model_on_video(video_path, model_path='best_bombus_model.pth'):
    """Simple test of your model on a video"""
    
    print(f"🎬 Testing model on: {Path(video_path).name}")
    
    # Load model
    model = BombusClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Image preprocessing
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
    
    # Test on 20 frames spaced throughout the video
    test_frames = np.linspace(0, total_frames-1, 20, dtype=int)
    
    bombus_detections = 0
    confidences = []
    
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
            
            if is_bombus:
                bombus_detections += 1
                timestamp = frame_idx / fps if fps > 0 else 0
                print(f"🐝 BOMBUS DETECTED at {timestamp:.1f}s (confidence: {conf_score:.3f})")
            
            confidences.append(conf_score if is_bombus else 0)
    
    cap.release()
    
    # Summary
    detection_rate = bombus_detections / len(test_frames)
    avg_confidence = np.mean([c for c in confidences if c > 0]) if bombus_detections > 0 else 0
    
    print(f"\n📈 ANALYSIS RESULTS:")
    print(f"   Frames analyzed: {len(test_frames)}")
    print(f"   Bombus detections: {bombus_detections}")
    print(f"   Detection rate: {detection_rate:.1%}")
    if bombus_detections > 0:
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"✅ Bombus activity detected in this video!")
    else:
        print(f"📹 No bombus activity detected.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python3 simple_video_test.py /path/to/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    test_model_on_video(video_path)