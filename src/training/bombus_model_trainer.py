"""
Video Analysis Script - Deploy Trained Bombus Detection Model
Author: Katerina Bischel
Project: Endangered Coastal Plant Pollinator Monitoring

This script uses trained model to analyze new videos for bombus activity.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BombusClassifier(nn.Module):
    """CNN model for bombus detection - must match your trained model architecture"""
    
    def __init__(self, num_classes=2):
        super(BombusClassifier, self).__init__()
        
        # Custom CNN architecture (matches your trained model)
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fifth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
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

class VideoAnalyzer:
    """Analyzes videos using your trained bombus detection model"""
    
    def __init__(self, model_path='best_bombus_model.pth', device='cpu'):
        self.device = device
        
        # Load the trained model
        self.model = BombusClassifier(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)
        
        # Image preprocessing (must match training transforms)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Model loaded successfully on {device}")
    
    def analyze_frame(self, frame):
        """
        Analyze a single frame for bombus presence
        Returns: (prediction, confidence_score)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply transforms
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction and confidence
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()
    
    def analyze_video(self, video_path, output_dir='analysis_results', 
                     frames_to_analyze=50, confidence_threshold=0.7):
        """
        Analyze an entire video for bombus activity
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save results
            frames_to_analyze: Number of frames to extract and analyze
            confidence_threshold: Minimum confidence for positive detection
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        logger.info(f"Analyzing video: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video properties: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
            
            # Select frames to analyze (evenly spaced)
            frame_indices = np.linspace(0, total_frames-1, min(frames_to_analyze, total_frames), dtype=int)
            
            # Analysis results
            detections = []
            bombus_frames = 0
            high_confidence_detections = 0
            
            for i, frame_idx in enumerate(frame_indices):
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Analyze frame
                prediction, confidence = self.analyze_frame(frame)
                timestamp = frame_idx / fps if fps > 0 else 0
                
                is_bombus = prediction == 1
                is_high_confidence = confidence > confidence_threshold
                
                if is_bombus:
                    bombus_frames += 1
                    if is_high_confidence:
                        high_confidence_detections += 1
                
                detections.append({
                    'frame_number': frame_idx,
                    'timestamp': timestamp,
                    'prediction': 'bombus' if is_bombus else 'no_bombus',
                    'confidence': confidence,
                    'high_confidence': is_high_confidence
                })
                
                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"Analyzed {i+1}/{len(frame_indices)} frames...")
            
            # Calculate summary statistics
            total_analyzed = len(detections)
            detection_rate = bombus_frames / total_analyzed if total_analyzed > 0 else 0
            high_conf_rate = high_confidence_detections / total_analyzed if total_analyzed > 0 else 0
            
            # Create analysis summary
            analysis_summary = {
                'video_file': str(video_path),
                'analysis_date': datetime.now().isoformat(),
                'video_properties': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'duration_seconds': duration
                },
                'analysis_settings': {
                    'frames_analyzed': total_analyzed,
                    'confidence_threshold': confidence_threshold
                },
                'results': {
                    'frames_with_bombus': bombus_frames,
                    'high_confidence_detections': high_confidence_detections,
                    'detection_rate': detection_rate,
                    'high_confidence_rate': high_conf_rate,
                    'avg_confidence': np.mean([d['confidence'] for d in detections if d['prediction'] == 'bombus']) if bombus_frames > 0 else 0
                },
                'detections': detections
            }
            
            # Save detailed results
            results_file = output_dir / f"{video_path.stem}_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(analysis_summary, f, indent=2)
            
            # Save CSV for easy viewing
            df = pd.DataFrame(detections)
            csv_file = output_dir / f"{video_path.stem}_detections.csv"
            df.to_csv(csv_file, index=False)
            
            # Create visualization
            self.create_detection_plot(analysis_summary, output_dir / f"{video_path.stem}_plot.png")
            
            # Print summary
            self.print_analysis_summary(analysis_summary)
            
            return analysis_summary
            
        finally:
            cap.release()
    
    def create_detection_plot(self, analysis_summary, output_path):
        """Create a visualization of detection results"""
        
        detections = analysis_summary['detections']
        
        # Extract data for plotting
        timestamps = [d['timestamp'] for d in detections]
        confidences = [d['confidence'] if d['prediction'] == 'bombus' else 0 for d in detections]
        bombus_times = [d['timestamp'] for d in detections if d['prediction'] == 'bombus']
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Timeline plot
        ax1.plot(timestamps, confidences, 'b-', alpha=0.7, label='Bombus Confidence')
        ax1.scatter(bombus_times, [max(confidences)] * len(bombus_times), 
                   color='red', s=50, alpha=0.8, label='Bombus Detected')
        ax1.axhline(y=analysis_summary['analysis_settings']['confidence_threshold'], 
                   color='orange', linestyle='--', label='Confidence Threshold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Confidence Score')
        ax1.set_title(f"Bombus Detection Timeline - {Path(analysis_summary['video_file']).name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Summary statistics
        results = analysis_summary['results']
        stats_text = f"""Detection Summary:
        
Frames Analyzed: {analysis_summary['analysis_settings']['frames_analyzed']}
Bombus Detections: {results['frames_with_bombus']}
Detection Rate: {results['detection_rate']:.1%}
High Confidence: {results['high_confidence_detections']}
Avg Confidence: {results['avg_confidence']:.3f}

Video Duration: {analysis_summary['video_properties']['duration_seconds']:.1f}s
Total Frames: {analysis_summary['video_properties']['total_frames']}
"""
        
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Analysis Summary')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detection plot saved to: {output_path}")
    
    def print_analysis_summary(self, analysis_summary):
        """Print a formatted summary of the analysis"""
        
        results = analysis_summary['results']
        video_name = Path(analysis_summary['video_file']).name
        
        print(f"\n" + "="*60)
        print(f"🎬 VIDEO ANALYSIS COMPLETE: {video_name}")
        print("="*60)
        
        print(f"📊 Detection Results:")
        print(f"   Frames analyzed: {analysis_summary['analysis_settings']['frames_analyzed']}")
        print(f"   Bombus detections: {results['frames_with_bombus']}")
        print(f"   Detection rate: {results['detection_rate']:.1%}")
        print(f"   High confidence detections: {results['high_confidence_detections']}")
        
        if results['frames_with_bombus'] > 0:
            print(f"   Average confidence: {results['avg_confidence']:.3f}")
            print(f"\n🐝 Bombus activity detected! This video contains pollinator visits.")
        else:
            print(f"\n📹 No bombus activity detected in analyzed frames.")
        
        print(f"\n📁 Results saved to analysis_results/")

def main():
    """Main analysis interface"""
    
    parser = argparse.ArgumentParser(description='Analyze videos for bombus activity')
    parser.add_argument('--video', required=True, help='Path to video file to analyze')
    parser.add_argument('--model', default='best_bombus_model.pth', help='Path to trained model')
    parser.add_argument('--output', default='analysis_results', help='Output directory')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames to analyze')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold')
    
    args = parser.parse_args()
    
    print("🐝 Bombus Video Analyzer")
    print("="*40)
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        print("💡 Make sure you've trained a model first using bombus_model_trainer.py")
        return
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        return
    
    try:
        # Initialize analyzer
        analyzer = VideoAnalyzer(model_path=args.model, device='cpu')
        
        # Analyze video
        results = analyzer.analyze_video(
            video_path=args.video,
            output_dir=args.output,
            frames_to_analyze=args.frames,
            confidence_threshold=args.threshold
        )
        
        if results:
            print(f"\n✅ Analysis completed successfully!")
            print(f"📈 Check {args.output}/ for detailed results and visualizations")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()