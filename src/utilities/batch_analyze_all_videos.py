"""
Comprehensive Batch Video Analysis for Conservation
Processes all videos and creates management reports
"""

import os
import subprocess
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

def batch_process_all_videos():
    """Process all videos in your collection"""
    
    # Your video base path
    base_path = "/Volumes/Expansion/summer2025_ncos_kb_collections"
    
    print("🎬 COMPREHENSIVE VIDEO ANALYSIS")
    print("="*50)
    
    # Find all videos
    video_extensions = ['.mp4', '.MP4', '.avi', '.mov']
    all_videos = []
    
    for ext in video_extensions:
        videos = list(Path(base_path).rglob(f'*{ext}'))
        all_videos.extend(videos)
    
    print(f"📹 Found {len(all_videos)} total videos")
    
    # Create results directory
    results_dir = Path("deployment_results")
    results_dir.mkdir(exist_ok=True)
    
    # Process each video
    analysis_results = []
    
    for i, video_path in enumerate(all_videos):
        print(f"\n🔍 Analyzing {i+1}/{len(all_videos)}: {video_path.name}")
        
        try:
            # Run analysis
            cmd = f'python3 src/correct_video_analyzer.py "{video_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Parse results (simplified)
            success = result.returncode == 0
            
            # Extract metadata from path
            path_parts = video_path.parts
            metadata = {
                'video_file': video_path.name,
                'full_path': str(video_path),
                'week': 'unknown',
                'site': 'unknown', 
                'shift': 'unknown',
                'analysis_success': success,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to parse metadata from path
            for part in path_parts:
                if 'week_' in part.lower():
                    metadata['week'] = part
                elif 'site_' in part.lower():
                    metadata['site'] = part
                elif part.lower() in ['morning', 'mid', 'afternoon']:
                    metadata['shift'] = part
            
            analysis_results.append(metadata)
            
        except Exception as e:
            print(f"❌ Error processing {video_path.name}: {e}")
            continue
    
    # Save batch results
    results_df = pd.DataFrame(analysis_results)
    results_df.to_csv(results_dir / "batch_analysis_summary.csv", index=False)
    
    print(f"\n✅ Batch analysis complete!")
    print(f"📊 Results saved to: {results_dir}/")
    print(f"   Successfully processed: {len([r for r in analysis_results if r['analysis_success']])} videos")
    
    return analysis_results

if __name__ == "__main__":
    batch_process_all_videos()
