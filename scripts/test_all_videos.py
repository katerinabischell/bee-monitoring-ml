#!/usr/bin/env python3
"""
Test YOLOv8 model on multiple Ventura Milkvetch videos - Expanded Analysis
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
        self.results_dir = self.data_dir / "results" / "video_analysis_expanded"
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Video directory - make it configurable for expanded analysis
        if video_base_dir:
            self.video_base_dir = Path(video_base_dir)
        else:
            self.video_base_dir = Path("/Volumes/Expansion1/summer2025_ncos_kb_collections/ventura_milkvetch")
            print("⚠️ Using default video path - consider passing video_base_dir parameter")
        
        print(f"🎬 VENTURA VIDEO TESTER - EXPANDED ANALYSIS")
        print(f"Models directory: {self.models_dir}")
        print(f"Results directory: {self.results_dir}")
        print(f"Video source: {self.video_base_dir}")
        print(f"Analyzing multiple weeks/days/sites")
    
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
        """Discover all MP4 videos in the expanded directory structure"""
        videos = []
        
        if not self.video_base_dir.exists():
            print(f"❌ Video base directory not found: {self.video_base_dir}")
            return videos
        
        time_periods = ['morning', 'mid', 'afternoon']
        
        # Look for week/day/site structure
        for item in self.video_base_dir.iterdir():
            if item.is_dir() and 'week' in item.name.lower():
                week_dir = item
                print(f"\n📁 Exploring {week_dir.name}")
                
                # Look for day directories
                for day_item in week_dir.iterdir():
                    if day_item.is_dir() and 'day' in day_item.name.lower():
                        day_dir = day_item
                        print(f"   📁 Found {day_dir.name}")
                        
                        # Look for site directories
                        for site_item in day_dir.iterdir():
                            if site_item.is_dir() and 'site' in site_item.name.lower():
                                site_dir = site_item
                                print(f"      📁 Found {site_dir.name}")
                                
                                # Look for time period directories
                                for period in time_periods:
                                    period_dir = site_dir / period
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
                                                    
                                                    # Create full ID for tracking
                                                    full_id = f"{week_dir.name}_{day_dir.name}_{site_dir.name}_{period}"
                                                    
                                                    videos.append({
                                                        'path': video_path,
                                                        'name': video_path.name,
                                                        'period': period,
                                                        'week': week_dir.name,
                                                        'day': day_dir.name,
                                                        'site': site_dir.name,
                                                        'full_id': full_id,
                                                        'duration_minutes': duration_minutes,
                                                        'fps': fps,
                                                        'frame_count': frame_count
                                                    })
                                                else:
                                                    print(f"⚠️ Could not open video: {video_path}")
                                            except Exception as e:
                                                print(f"❌ Error processing video {video_path}: {e}")
                                                continue
                                        
                                        if mp4_files:
                                            print(f"         Found {len(mp4_files)} videos in {period}")
                                    else:
                                        print(f"         ⚠️ No {period} directory found")
        
        print(f"\n📊 Total videos discovered: {len(videos)}")
        return videos
    
    def analyze_single_video(self, video_info, model, sample_interval=30):
        """Analyze a single video for bee detections"""
        print(f"\n🔍 Analyzing: {video_info['full_id']} - {video_info['name']} ({video_info['duration_minutes']:.1f} min)")
        
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
                            'period': video_info['period'],
                            'week': video_info['week'],
                            'day': video_info['day'],
                            'site': video_info['site'],
                            'full_id': video_info['full_id']
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
        """Create comprehensive analysis summary for expanded dataset"""
        print(f"\n📊 CREATING EXPANDED VIDEO ANALYSIS SUMMARY")
        
        if not all_detections:
            print("❌ No detections to summarize!")
            return None, None, {}
        
        df = pd.DataFrame(all_detections)
        
        # Create summary statistics by different groupings
        summary_stats = {}
        
        # By period
        for period in df['period'].unique():
            period_data = df[df['period'] == period]
            summary_stats[f"period_{period}"] = {
                'total_samples': len(period_data),
                'samples_with_bees': len(period_data[period_data['num_bees'] > 0]),
                'total_bee_detections': period_data['num_bees'].sum(),
                'avg_bees_per_sample': period_data['num_bees'].mean(),
                'max_bees_single_sample': period_data['num_bees'].max(),
                'avg_confidence': period_data[period_data['num_bees'] > 0]['avg_confidence'].mean() if len(period_data[period_data['num_bees'] > 0]) > 0 else 0
            }
        
        # By week (if multiple weeks exist)
        if 'week' in df.columns and df['week'].nunique() > 1:
            for week in df['week'].unique():
                week_data = df[df['week'] == week]
                summary_stats[f"week_{week}"] = {
                    'total_samples': len(week_data),
                    'samples_with_bees': len(week_data[week_data['num_bees'] > 0]),
                    'total_bee_detections': week_data['num_bees'].sum(),
                    'avg_bees_per_sample': week_data['num_bees'].mean(),
                    'max_bees_single_sample': week_data['num_bees'].max(),
                    'avg_confidence': week_data[week_data['num_bees'] > 0]['avg_confidence'].mean() if len(week_data[week_data['num_bees'] > 0]) > 0 else 0
                }
        
        # Create expanded visualizations
        try:
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Bee detections over time by period
            ax1 = plt.subplot(3, 3, 1)
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
            ax2 = plt.subplot(3, 3, 2)
            bee_counts = df['num_bees'].value_counts().sort_index()
            ax2.bar(bee_counts.index, bee_counts.values, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Number of Bees per Sample')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Bee Counts')
            ax2.grid(True, alpha=0.3)
            
            # 3. Activity by time period
            ax3 = plt.subplot(3, 3, 3)
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
            
            # 4. Activity by week (if multiple weeks)
            if 'week' in df.columns and df['week'].nunique() > 1:
                ax4 = plt.subplot(3, 3, 4)
                week_summary = df.groupby('week')['num_bees'].agg(['sum', 'mean']).reset_index()
                ax4.bar(range(len(week_summary)), week_summary['sum'], alpha=0.7)
                ax4.set_xlabel('Week')
                ax4.set_ylabel('Total Bee Detections')
                ax4.set_title('Total Activity by Week')
                ax4.set_xticks(range(len(week_summary)))
                ax4.set_xticklabels(week_summary['week'], rotation=45)
                ax4.grid(True, alpha=0.3)
            
            # 5. Confidence distribution
            ax5 = plt.subplot(3, 3, 5)
            conf_data = df[df['num_bees'] > 0]['avg_confidence']
            if len(conf_data) > 0:
                ax5.hist(conf_data, bins=20, alpha=0.7, edgecolor='black')
                ax5.set_xlabel('Average Confidence')
                ax5.set_ylabel('Frequency')
                ax5.set_title('Confidence Score Distribution')
                ax5.grid(True, alpha=0.3)
            
            # 6. Heatmap by day and period (if multiple days)
            if 'day' in df.columns and df['day'].nunique() > 1:
                ax6 = plt.subplot(3, 3, 6)
                heatmap_data = df.groupby(['day', 'period'])['num_bees'].sum().unstack(fill_value=0)
                im = ax6.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
                ax6.set_xticks(range(len(heatmap_data.columns)))
                ax6.set_xticklabels(heatmap_data.columns)
                ax6.set_yticks(range(len(heatmap_data.index)))
                ax6.set_yticklabels(heatmap_data.index)
                ax6.set_title('Activity Heatmap: Day vs Period')
                plt.colorbar(im, ax=ax6)
            
            # 7-9. Summary statistics tables
            ax7 = plt.subplot(3, 3, 7)
            stats_text = "PERIOD STATISTICS:\n\n"
            for key, stats in summary_stats.items():
                if key.startswith('period_'):
                    period = key.replace('period_', '')
                    stats_text += f"{period.upper()}:\n"
                    stats_text += f"  Samples: {stats['total_samples']}\n"
                    stats_text += f"  With bees: {stats['samples_with_bees']}\n"
                    stats_text += f"  Total: {stats['total_bee_detections']}\n"
                    stats_text += f"  Avg: {stats['avg_bees_per_sample']:.2f}\n"
                    stats_text += f"  Conf: {stats['avg_confidence']:.2f}\n\n"
            
            ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
            ax7.set_title('Period Statistics')
            ax7.axis('off')
            
            plt.suptitle('Ventura Milkvetch Bee Detection - Expanded Analysis Summary', fontsize=16)
            plt.tight_layout()
            
            # Save summary
            summary_path = self.results_dir / 'expanded_video_analysis_summary.jpg'
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Error creating plots: {e}")
            summary_path = None
        
        # Save data to CSV
        try:
            csv_path = self.results_dir / 'expanded_detection_results.csv'
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
        print(f"\n🖼️ Creating sample detections for {video_info['full_id']} - {video_info['name']}")
        
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
                        
                        # Add timestamp and ID
                        timestamp = frame_num / video_info['fps']
                        timestamp_text = f"Time: {int(timestamp//60):02d}:{int(timestamp%60):02d}"
                        id_text = f"ID: {video_info['full_id']}"
                        
                        cv2.putText(annotated_frame, timestamp_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, id_text, (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Save sample
                        sample_name = f"sample_{video_info['full_id']}_frame_{frame_num}.jpg"
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
        """Run complete expanded video analysis pipeline"""
        print(f"\n🎬 STARTING EXPANDED VIDEO ANALYSIS")
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
            print(f"   {video['full_id']}: {video['name']} ({video['duration_minutes']:.1f} min)")
        
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
        
        print(f"\n✅ EXPANDED VIDEO ANALYSIS COMPLETE!")
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
    """Run expanded video analysis"""
    # Check requirements
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed. Install with:")
        print("pip install ultralytics")
        return
    
    # Start analysis
    tester = VenturaVideoTester()
    
    # Run full analysis with improved threshold
    results = tester.run_full_video_analysis(
        sample_interval=30,      # Sample every 30 seconds
        confidence_threshold=0.4 # Improved threshold based on verification
    )
    
    if results:
        print("\n🎉 Analysis complete! Check results in data/results/video_analysis_expanded/")
        print("View summary plot and CSV data for detailed insights across multiple weeks/days/sites")

if __name__ == "__main__":
    main()