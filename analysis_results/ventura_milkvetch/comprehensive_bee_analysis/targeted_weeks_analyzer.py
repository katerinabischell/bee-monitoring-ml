#!/usr/bin/env python3
"""
 Analyze Week 4 and Week 5 Birds Beak videos specifically
"""

import os
from pathlib import Path
from production_bee_analyzer import analyze_video_with_yolo, generate_combined_report
import json
from datetime import datetime

def analyze_specific_weeks():
    """
    Analyze Week 4 and Week 5 Birds Beak videos
    """
    
    model_path = "models/object_detection/ventura_bombus_yolo/weights/best.pt"
    base_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/birds_beak"
    
    # Target directories
    target_paths = [
        f"{base_path}/week_4",
        f"{base_path}/week 4",  # Try with and without underscore
        f"{base_path}/week_5", 
        f"{base_path}/week 5"
    ]
    
    print("🐝 WEEK 4-5 BIRDS BEAK ANALYSIS")
    print("="*60)
    print("Analyzing Birds Beak videos from Week 4 and Week 5")
    print("="*60)
    
    # Find target videos
    target_videos = []
    
    for target_path in target_paths:
        if os.path.exists(target_path):
            print(f"✅ Found directory: {target_path}")
            
            # Find all MP4 files in this week
            week_path = Path(target_path)
            videos = list(week_path.rglob("*.MP4")) + list(week_path.rglob("*.mp4"))
            
            for video in videos:
                target_videos.append(video)
                print(f"   📹 {video.relative_to(Path(base_path))}")
        else:
            print(f"⚠️ Directory not found: {target_path}")
    
    if not target_videos:
        print("❌ No videos found in Week 4-5 Birds Beak directories")
        print("💡 Let's check what directories actually exist:")
        
        birds_beak_path = Path(base_path)
        if birds_beak_path.exists():
            print(f"\nDirectories in {base_path}:")
            for item in sorted(birds_beak_path.iterdir()):
                if item.is_dir():
                    print(f"   📁 {item.name}")
        return None
    
    print(f"\n🎬 Found {len(target_videos)} videos to analyze")
    
    # Create output directory
    output_dir = Path("week4_5_birds_beak_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Analyze each video
    all_results = []
    
    for i, video_path in enumerate(target_videos, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(target_videos)}")
        print(f"📁 {video_path.relative_to(Path(base_path))}")
        
        results = analyze_video_with_yolo(str(video_path), model_path, confidence=0.1)
        
        if results:
            all_results.append(results)
            
            # Add metadata about week and path
            results['week_info'] = {
                'week': 4 if 'week_4' in str(video_path) or 'week 4' in str(video_path) else 5,
                'plant_type': 'birds_beak',
                'relative_path': str(video_path.relative_to(Path(base_path)))
            }
            
            # Save individual results
            safe_name = video_path.stem.replace(' ', '_')
            output_file = output_dir / f"{safe_name}_analysis.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Progress update
        if i % 5 == 0:
            print(f"📊 Progress: {i}/{len(target_videos)} videos processed")
    
    # Generate summary report
    if all_results:
        print(f"\n📈 GENERATING SUMMARY REPORT")
        
        # Week-by-week breakdown
        week4_results = [r for r in all_results if r.get('week_info', {}).get('week') == 4]
        week5_results = [r for r in all_results if r.get('week_info', {}).get('week') == 5]
        
        # Create comprehensive summary
        summary_report = {
            'analysis_date': datetime.now().isoformat(),
            'dataset': 'Week 4-5 Birds Beak',
            'total_videos': len(all_results),
            'model_used': model_path,
            'confidence_threshold': 0.1,
            'week_breakdown': {
                'week_4': {
                    'videos': len(week4_results),
                    'total_detections': sum(r['summary']['total_bee_detections'] for r in week4_results),
                    'videos_with_bees': len([r for r in week4_results if r['summary']['total_bee_detections'] > 0]),
                    'avg_detections': sum(r['summary']['total_bee_detections'] for r in week4_results) / len(week4_results) if week4_results else 0
                },
                'week_5': {
                    'videos': len(week5_results),
                    'total_detections': sum(r['summary']['total_bee_detections'] for r in week5_results),
                    'videos_with_bees': len([r for r in week5_results if r['summary']['total_bee_detections'] > 0]),
                    'avg_detections': sum(r['summary']['total_bee_detections'] for r in week5_results) / len(week5_results) if week5_results else 0
                }
            },
            'overall_stats': {
                'total_detections': sum(r['summary']['total_bee_detections'] for r in all_results),
                'videos_with_activity': len([r for r in all_results if r['summary']['total_bee_detections'] > 0]),
                'detection_rate': len([r for r in all_results if r['summary']['total_bee_detections'] > 0]) / len(all_results),
                'avg_detections_per_video': sum(r['summary']['total_bee_detections'] for r in all_results) / len(all_results)
            },
            'top_videos': sorted(all_results, key=lambda x: x['summary']['total_bee_detections'], reverse=True)[:5],
            'individual_results': all_results
        }
        
        # Save summary
        with open(output_dir / "week4_5_summary_report.json", 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📊 WEEK 4-5 BIRDS BEAK ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total videos analyzed: {len(all_results)}")
        print(f"")
        print(f"Week 4 Results:")
        print(f"   Videos: {summary_report['week_breakdown']['week_4']['videos']}")
        print(f"   Total detections: {summary_report['week_breakdown']['week_4']['total_detections']}")
        print(f"   Videos with bees: {summary_report['week_breakdown']['week_4']['videos_with_bees']}")
        print(f"   Avg per video: {summary_report['week_breakdown']['week_4']['avg_detections']:.1f}")
        print(f"")
        print(f"Week 5 Results:")
        print(f"   Videos: {summary_report['week_breakdown']['week_5']['videos']}")
        print(f"   Total detections: {summary_report['week_breakdown']['week_5']['total_detections']}")
        print(f"   Videos with bees: {summary_report['week_breakdown']['week_5']['videos_with_bees']}")
        print(f"   Avg per video: {summary_report['week_breakdown']['week_5']['avg_detections']:.1f}")
        print(f"")
        print(f"📁 Results saved to: {output_dir}")
        
        return summary_report
    else:
        print("❌ No videos successfully analyzed")
        return None

def main():
    """Run the Week 4-5 analysis"""
    analyze_specific_weeks()

if __name__ == "__main__":
    main()