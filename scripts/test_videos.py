#!/usr/bin/env python3
"""
Test YOLOv8 model on multiple Ventura Milkvetch videos
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class VenturaVideoTester:
    def __init__(self, data_dir="data", video_base_dir=None):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.results_dir = self.data_dir / "results" / "video_analysis"
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Video directory - make it configurable
        if video_base_dir:
            self.video_base_dir = Path(video_base_dir)
        else:
            self.video_base_dir = Path("/Volumes/Expansion1/summer2025_ncos_kb_collections/ventura_milkvetch/week 6/day 1/site 1")
            print("⚠️ Using default video path - consider passing video_base_dir parameter")
        
        print(f"🎬 VENTURA VIDEO TESTER")
        print(f"Models directory: {self.models_dir}")
        print(f"Results directory: {self.results_dir}")
        print(f"Video source: {self.video_base_dir}")
    
    def find_trained_model(self):
        """Find the best trained model"""
        # Try the new improved model first
        model_path_v2 = self.models_dir / 'ventura_bee_detection_v2' / 'weights' / 'best.pt'
        model_path_v1 = self.models_dir / 'ventura_bee_detection' / 'weights' / 'best.pt'
        
        if model_path_v2.exists():
            print(f"✅ Found improved model: {model_path_v2}")
            print(f"   Model metrics: mAP50=0.745, Precision=0.878, Recall=0.648")
            return model_path_v2
        elif model_path_v1.exists():
            print(f"✅ Found original model: {model_path_v1}")
            print(f"   Model metrics: mAP50=0.454, Precision=0.729, Recall=0.406")
            return model_path_v1
        else:
            print(f"❌ No trained model found")
            print("💡 Run 3_train_model.py first")
            return None
    
    def discover_videos(self):
        """Discover all MP4 videos in the specified directory structure"""
        videos = []
        
        # Check if base directory exists
        if not self.video_base_dir.exists():
            print(f"❌ Video base directory not found: {self.video_base_dir}")
            return videos
        
        time_periods = ['morning', 'mid', 'afternoon']
        
        for period in time_periods:
            period_dir = self.video_base_dir / period
            if period_dir.exists():
                # Look for both .MP4 and .mp4 files
                mp4_files = list(period_dir.glob('*.MP4')) + list(period_dir.glob('*.mp4'))
                
                for video_path in mp4_files:
                    try:
                        # Get video info
                        cap = cv2.VideoCapture(str(video_path))
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                            duration_minutes = (frame_count / fps) / 60 if fps > 0 else 0
                            cap.release()
                            
                            videos.append({
                                'path': video_path,
                                'name': video_path.name,
                                'period': period,
                                'duration_minutes': duration_minutes,
                                'fps': fps,
                                'frame_count': frame_count
                            })
                        else:
                            print(f"⚠️ Could not open video: {video_path}")
                    except Exception as e:
                        print(f"❌ Error processing video {video_path}: {e}")
                        continue
                    
                print(f"Found {len(mp4_files)} videos in {period}")
            else:
                print(f"⚠️ Directory not found: {period_dir}")
        
        return videos
    
    def analyze_single_video(self, video_info, model, sample_interval=30):
        """Analyze a single video for bee detections"""
        print(f"\n🔍 Analyzing: {video_info['name']} ({video_info['duration_minutes']:.1f} min)")
        
        video_path = video_info['path']
        detections = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"❌ Failed to open video: {video_path}")
                return detections
            
            frame_number = 0
            total_frames = int(video_info['frame_count'])
            fps = video_info['fps']
            
            # Sample frames at specified interval (seconds)
            sample_frame_interval = int(fps * sample_interval)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every sample_interval seconds
                if frame_number % sample_frame_interval == 0:
                    timestamp_seconds = frame_number / fps
                    timestamp_minutes = timestamp_seconds / 60
                    
                    try:
                        # Run detection
                        results = model(frame)
                        
                        # Extract detections
                        boxes = results[0].boxes
                        num_bees = len(boxes) if boxes is not None else 0
                        
                        detection_data = {
                            'frame_number': frame_number,
                            'timestamp_seconds': timestamp_seconds,
                            'timestamp_minutes': timestamp_minutes,
                            'timestamp_formatted': f"{int(timestamp_minutes):02d}:{int(timestamp_seconds % 60):02d}",
                            'num_bees': num_bees,
                            'video_name': video_info['name'],
                            'period': video_info['period']
                        }
                        
                        # Add confidence scores if bees detected
                        if boxes is not None and len(boxes) > 0:
                            confidences = boxes.conf.cpu().numpy()
                            if len(confidences) > 0:  # Extra safety check
                                detection_data['max_confidence'] = float(np.max(confidences))
                                detection_data['avg_confidence'] = float(np.mean(confidences))
                                detection_data['min_confidence'] = float(np.min(confidences))
                            else:
                                detection_data['max_confidence'] = 0.0
                                detection_data['avg_confidence'] = 0.0
                                detection_data['min_confidence'] = 0.0
                        else:
                            detection_data['max_confidence'] = 0.0
                            detection_data['avg_confidence'] = 0.0
                            detection_data['min_confidence'] = 0.0
                        
                        detections.append(detection_data)
                        
                        if frame_number % (sample_frame_interval * 10) == 0:  # Progress every 10 samples
                            progress = (frame_number / total_frames) * 100
                            print(f"   Progress: {progress:.1f}% - Frame {frame_number}/{total_frames}")
                    
                    except Exception as e:
                        print(f"⚠️ Error processing frame {frame_number}: {e}")
                        continue
                
                frame_number += 1
            
        except Exception as e:
            print(f"❌ Error analyzing video {video_path}: {e}")
        finally:
            if 'cap' in locals():
                cap.release()
        
        print(f"✅ Completed analysis: {len(detections)} samples")
        return detections
    
    def create_video_summary(self, all_detections):
        """Create comprehensive analysis summary"""
        print(f"\n📊 CREATING VIDEO ANALYSIS SUMMARY")
        
        if not all_detections:
            print("❌ No detections to summarize!")
            return None, None, {}
        
        df = pd.DataFrame(all_detections)
        
        # Create summary statistics
        summary_stats = {}
        
        for period in df['period'].unique():
            period_data = df[df['period'] == period]
            
            summary_stats[period] = {
                'total_samples': len(period_data),
                'samples_with_bees': len(period_data[period_data['num_bees'] > 0]),
                'total_bee_detections': period_data['num_bees'].sum(),
                'avg_bees_per_sample': period_data['num_bees'].mean(),
                'max_bees_single_sample': period_data['num_bees'].max(),
                'avg_confidence': period_data[period_data['num_bees'] > 0]['avg_confidence'].mean() if len(period_data[period_data['num_bees'] > 0]) > 0 else 0
            }
        
        # Create visualizations
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Ventura Milkvetch Bee Detection Analysis Summary', fontsize=16)
            
            # 1. Bee detections over time by period
            for period in df['period'].unique():
                period_data = df[df['period'] == period]
                ax1.plot(period_data['timestamp_minutes'], period_data['num_bees'], 
                        marker='o', markersize=3, label=f'{period.title()}', alpha=0.7)
            
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Number of Bees Detected')
            ax1.set_title('Bee Activity Over Time by Period')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Detection count distribution
            bee_counts = df['num_bees'].value_counts().sort_index()
            ax2.bar(bee_counts.index, bee_counts.values, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Number of Bees per Sample')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Bee Counts')
            ax2.grid(True, alpha=0.3)
            
            # 3. Activity by time period
            period_summary = df.groupby('period')['num_bees'].agg(['sum', 'mean', 'count']).reset_index()
            x_pos = range(len(period_summary))
            ax3.bar([x - 0.2 for x in x_pos], period_summary['sum'], 0.4, label='Total Detections', alpha=0.7)
            ax3.bar([x + 0.2 for x in x_pos], period_summary['mean'], 0.4, label='Avg per Sample', alpha=0.7)
            ax3.set_xlabel('Time Period')
            ax3.set_ylabel('Bee Detections')
            ax3.set_title('Bee Activity by Time Period')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([p.title() for p in period_summary['period']])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Summary statistics table
            stats_text = "SUMMARY STATISTICS:\n\n"
            for period, stats in summary_stats.items():
                stats_text += f"{period.upper()}:\n"
                stats_text += f"  Total samples: {stats['total_samples']}\n"
                stats_text += f"  Samples with bees: {stats['samples_with_bees']}\n"
                stats_text += f"  Total detections: {stats['total_bee_detections']}\n"
                stats_text += f"  Avg bees/sample: {stats['avg_bees_per_sample']:.2f}\n"
                stats_text += f"  Max bees/sample: {stats['max_bees_single_sample']}\n"
                stats_text += f"  Avg confidence: {stats['avg_confidence']:.2f}\n\n"
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
            ax4.set_title('Summary Statistics')
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save summary
            summary_path = self.results_dir / 'video_analysis_summary.jpg'
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Error creating plots: {e}")
            summary_path = None
        
        # Save data to CSV
        try:
            csv_path = self.results_dir / 'detection_results.csv'
            df.to_csv(csv_path, index=False)
            print(f"✅ Data saved: {csv_path}")
        except Exception as e:
            print(f"⚠️ Error saving CSV: {e}")
            csv_path = None
        
        if summary_path:
            print(f"✅ Summary saved: {summary_path}")
        
        return summary_path, csv_path, summary_stats
    
    def create_sample_detections(self, video_info, model, num_samples=5):
        """Create sample detection images from video"""
        print(f"\n🖼️ Creating sample detections for {video_info['name']}")
        
        video_path = video_info['path']
        sample_images = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"❌ Failed to open video: {video_path}")
                return sample_images
            
            total_frames = int(video_info['frame_count'])
            sample_frames = np.linspace(0, total_frames-1, num_samples, dtype=int)
            
            for i, frame_num in enumerate(sample_frames):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Run detection
                        results = model(frame)
                        
                        # Draw detections
                        annotated_frame = results[0].plot()
                        
                        # Add timestamp
                        timestamp = frame_num / video_info['fps']
                        timestamp_text = f"Time: {int(timestamp//60):02d}:{int(timestamp%60):02d}"
                        cv2.putText(annotated_frame, timestamp_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Save sample
                        sample_name = f"sample_{video_info['period']}_{Path(video_info['name']).stem}_frame_{frame_num}.jpg"
                        sample_path = self.results_dir / sample_name
                        cv2.imwrite(str(sample_path), annotated_frame)
                        sample_images.append(sample_path)
                    else:
                        print(f"⚠️ Could not read frame {frame_num}")
                except Exception as e:
                    print(f"⚠️ Error processing frame {frame_num}: {e}")
                    continue
        
        except Exception as e:
            print(f"❌ Error creating samples from {video_path}: {e}")
        finally:
            if 'cap' in locals():
                cap.release()
        
        print(f"✅ Created {len(sample_images)} sample detection images")
        return sample_images
    
    def run_full_video_analysis(self, sample_interval=30, confidence_threshold=0.25):
        """Run complete video analysis pipeline"""
        print(f"\n🎬 STARTING FULL VIDEO ANALYSIS")
        print(f"{'='*60}")
        print(f"Sample interval: {sample_interval} seconds")
        print(f"Confidence threshold: {confidence_threshold}")
        
        # Find model
        model_path = self.find_trained_model()
        if not model_path:
            return None
        
        try:
            # Load model
            model = YOLO(model_path)
            model.conf = confidence_threshold  # Set confidence threshold
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
        
        # Discover videos
        videos = self.discover_videos()
        if not videos:
            print("❌ No videos found!")
            return None
        
        print(f"\n📹 Found {len(videos)} videos to analyze:")
        for video in videos:
            print(f"   {video['period']}: {video['name']} ({video['duration_minutes']:.1f} min)")
        
        # Analyze each video
        all_detections = []
        sample_images = []
        
        for video_info in videos:
            # Analyze video
            detections = self.analyze_single_video(video_info, model, sample_interval)
            all_detections.extend(detections)
            
            # Create sample images
            samples = self.create_sample_detections(video_info, model, num_samples=3)
            sample_images.extend(samples)
        
        # FIXED: Call the correct method name
        summary_path, csv_path, stats = self.create_video_summary(all_detections)
        
        print(f"\n✅ VIDEO ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"📊 Total samples analyzed: {len(all_detections)}")
        print(f"🐝 Total bee detections: {sum(d['num_bees'] for d in all_detections)}")
        print(f"🖼️ Sample images: {len(sample_images)}")
        if summary_path:
            print(f"📈 Summary plot: {summary_path}")
        if csv_path:
            print(f"📄 Data CSV: {csv_path}")
        
        return {
            'detections': all_detections,
            'summary_path': summary_path,
            'csv_path': csv_path,
            'sample_images': sample_images,
            'stats': stats
        }


def main():
    """Run video analysis"""
    # Check requirements
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed. Install with:")
        print("pip install ultralytics")
        return
    
    # Start analysis
    tester = VenturaVideoTester()
    
    # Run full analysis
    results = tester.run_full_video_analysis(
        sample_interval=30,      # Sample every 30 seconds
        confidence_threshold=0.4
    )
    
    if results:
        print("Analysis complete! Check results in data/results/video_analysis/")
        print("View summary plot and CSV data for detailed insights")

if __name__ == "__main__":
    main()