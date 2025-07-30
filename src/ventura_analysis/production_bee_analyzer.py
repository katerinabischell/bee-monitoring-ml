#!/usr/bin/env python3
"""
Production bee video analyzer using your trained YOLO model
Optimized for research use with confidence threshold 0.1
"""

import os
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

def analyze_video_with_yolo(video_path, model_path, confidence=0.1, segment_duration=300):
    """
    Analyze video for bee activity using trained YOLO model
    
    Args:
        video_path: Path to video file
        model_path: Path to trained YOLO model
        confidence: Detection confidence threshold (0.1 recommended)
        segment_duration: Length of segments in seconds (default 5 minutes)
    
    Returns:
        Dictionary with analysis results
    """
    
    try:
        from ultralytics import YOLO
        
        # Load trained model
        model = YOLO(model_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"🎬 Analyzing: {Path(video_path).name}")
        print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"   FPS: {fps:.1f}")
        print(f"   Confidence threshold: {confidence}")
        
        # Analyze in segments
        num_segments = int(duration // segment_duration) + 1
        segments = []
        
        for segment_num in range(num_segments):
            start_time = segment_num * segment_duration
            end_time = min((segment_num + 1) * segment_duration, duration)
            
            print(f"   📊 Segment {segment_num + 1}: {start_time:.0f}-{end_time:.0f}s")
            
            # Sample frames every 10 seconds within segment
            segment_detections = []
            
            for timestamp in range(int(start_time), int(end_time), 10):
                frame_num = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    results = model(frame, conf=confidence, verbose=False)
                    bee_count = len(results[0].boxes) if results[0].boxes is not None else 0
                    
                    # Get confidence scores
                    confidences = results[0].boxes.conf.tolist() if results[0].boxes is not None else []
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    segment_detections.append({
                        'timestamp': timestamp,
                        'bee_count': bee_count,
                        'avg_confidence': avg_confidence,
                        'individual_confidences': confidences
                    })
            
            # Summarize segment
            total_bees = sum(d['bee_count'] for d in segment_detections)
            frames_with_bees = len([d for d in segment_detections if d['bee_count'] > 0])
            max_bees = max(d['bee_count'] for d in segment_detections) if segment_detections else 0
            
            segment_summary = {
                'segment_number': segment_num + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'total_detections': total_bees,
                'frames_with_bees': frames_with_bees,
                'max_simultaneous_bees': max_bees,
                'detection_rate': frames_with_bees / len(segment_detections) if segment_detections else 0,
                'detections': segment_detections
            }
            
            segments.append(segment_summary)
            
            if total_bees > 0:
                print(f"      🐝 {total_bees} total detections, max {max_bees} simultaneous")
        
        cap.release()
        
        # Overall summary
        total_detections = sum(s['total_detections'] for s in segments)
        active_segments = len([s for s in segments if s['total_detections'] > 0])
        peak_segment = max(segments, key=lambda x: x['total_detections']) if segments else None
        
        analysis_results = {
            'video_path': video_path,
            'video_name': Path(video_path).name,
            'analysis_date': datetime.now().isoformat(),
            'model_used': model_path,
            'confidence_threshold': confidence,
            'video_info': {
                'duration_seconds': duration,
                'fps': fps,
                'total_frames': total_frames
            },
            'summary': {
                'total_segments': len(segments),
                'active_segments': active_segments,
                'total_bee_detections': total_detections,
                'peak_segment': peak_segment['segment_number'] if peak_segment else None,
                'peak_detections': peak_segment['total_detections'] if peak_segment else 0,
                'overall_activity_rate': active_segments / len(segments) if segments else 0
            },
            'segments': segments
        }
        
        print(f"\n📊 ANALYSIS COMPLETE:")
        print(f"   Total bee detections: {total_detections}")
        print(f"   Active segments: {active_segments}/{len(segments)}")
        print(f"   Peak activity: Segment {peak_segment['segment_number']} ({peak_segment['total_detections']} bees)" if peak_segment else "   No peak activity")
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Error analyzing video: {e}")
        return None

def batch_analyze_videos(video_directory, model_path, output_dir="yolo_analysis_results"):
    """
    Analyze multiple videos and generate comprehensive report
    """
    
    video_dir = Path(video_directory)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all video files
    video_files = []
    for pattern in ['*.mp4', '*.MP4', '*.avi', '*.mov']:
        video_files.extend(video_dir.rglob(pattern))
    
    print(f"🔍 Found {len(video_files)} video files in {video_directory}")
    
    all_results = []
    
    for i, video_path in enumerate(video_files[:5], 1):  # Limit to first 5 for testing
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{min(5, len(video_files))}")
        
        results = analyze_video_with_yolo(str(video_path), model_path)
        
        if results:
            all_results.append(results)
            
            # Save individual video results
            video_output = output_dir / f"{results['video_name'].replace('.MP4', '')}_yolo_analysis.json"
            with open(video_output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
    
    # Generate combined report
    if all_results:
        combined_report = generate_combined_report(all_results, output_dir)
        print(f"\n📋 Analysis complete! Results saved to {output_dir}")
        return combined_report
    else:
        print("❌ No videos successfully analyzed")
        return None

def generate_combined_report(all_results, output_dir):
    """Generate combined analysis report"""
    
    # Aggregate statistics
    total_videos = len(all_results)
    total_detections = sum(r['summary']['total_bee_detections'] for r in all_results)
    videos_with_bees = len([r for r in all_results if r['summary']['total_bee_detections'] > 0])
    
    # Find peak videos
    videos_by_activity = sorted(all_results, key=lambda x: x['summary']['total_bee_detections'], reverse=True)
    
    combined_report = {
        'analysis_date': datetime.now().isoformat(),
        'total_videos_analyzed': total_videos,
        'videos_with_bee_activity': videos_with_bees,
        'total_bee_detections': total_detections,
        'average_detections_per_video': total_detections / total_videos if total_videos > 0 else 0,
        'most_active_videos': videos_by_activity[:3],
        'individual_results': all_results
    }
    
    # Save combined report
    with open(output_dir / "combined_yolo_analysis.json", 'w') as f:
        json.dump(combined_report, f, indent=2, default=str)
    
    print(f"\n📈 COMBINED ANALYSIS REPORT:")
    print(f"   Videos analyzed: {total_videos}")
    print(f"   Videos with bees: {videos_with_bees}")
    print(f"   Total detections: {total_detections}")
    print(f"   Avg per video: {total_detections/total_videos:.1f}")
    
    return combined_report

def main():
    """Main analysis function"""
    
    model_path = "models/object_detection/ventura_bombus_yolo/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🐝 PRODUCTION BEE VIDEO ANALYZER")
    print("="*60)
    print("Using trained YOLO model with optimized confidence threshold")
    print("="*60)
    
    # Test on your Ventura video first
    test_video = "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milkvetch/week 5/day 1/site 1/afternoon/P1000446.MP4"
    
    if os.path.exists(test_video):
        print("\n1️⃣ Testing on Ventura Milk Vetch video...")
        results = analyze_video_with_yolo(test_video, model_path, confidence=0.1)
        
        if results:
            # Save results
            with open("P1000446_production_analysis.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✅ Results saved to P1000446_production_analysis.json")
    
    # Option to analyze more videos
    print(f"\n💡 To analyze more videos:")
    print(f"   batch_analyze_videos('/path/to/video/directory', '{model_path}')")

if __name__ == "__main__":
    main()