"""
Deploy Real-Time Monitoring System
Sets up automated processing for ongoing video analysis
"""

import os
import schedule
import time
from pathlib import Path
import subprocess

def process_new_videos():
    """Check for and process any new videos"""
    
    print(f"🔍 Checking for new videos at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Your video directory
    video_base = "/Volumes/Expansion/summer2025_ncos_kb_collections"
    
    # Create a log of processed videos
    processed_log = Path("processed_videos.log")
    
    # Get list of already processed videos
    processed_videos = set()
    if processed_log.exists():
        with open(processed_log, 'r') as f:
            processed_videos = set(line.strip() for line in f)
    
    # Find new videos
    all_videos = list(Path(video_base).rglob('*.mp4'))
    new_videos = [v for v in all_videos if str(v) not in processed_videos]
    
    if new_videos:
        print(f"📹 Found {len(new_videos)} new videos to process")
        
        for video in new_videos:
            print(f"🎬 Processing: {video.name}")
            
            try:
                # Analyze video
                cmd = f'python3 src/correct_video_analyzer.py "{video}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Mark as processed
                    with open(processed_log, 'a') as f:
                        f.write(f"{video}\n")
                    print(f"✅ Successfully processed: {video.name}")
                else:
                    print(f"❌ Failed to process: {video.name}")
                    
            except Exception as e:
                print(f"❌ Error processing {video.name}: {e}")
    else:
        print("📹 No new videos found")

def setup_monitoring_schedule():
    """Set up automated monitoring schedule"""
    
    print("🤖 SETTING UP AUTOMATED MONITORING")
    print("="*50)
    
    # Schedule monitoring checks
    schedule.every(1).hours.do(process_new_videos)  # Check every hour
    schedule.every().day.at("09:00").do(process_new_videos)  # Daily morning check
    schedule.every().day.at("17:00").do(process_new_videos)  # Daily evening check
    
    print("⏰ Monitoring schedule configured:")
    print("   • Every hour: Check for new videos")
    print("   • 9:00 AM: Daily morning analysis")
    print("   • 5:00 PM: Daily evening analysis")
    print()
    print("🚀 Monitoring system active!")
    print("Press Ctrl+C to stop monitoring")
    
    # Run initial check
    process_new_videos()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    setup_monitoring_schedule()
