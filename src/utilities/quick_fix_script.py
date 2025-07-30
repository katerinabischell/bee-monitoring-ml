#!/usr/bin/env python3
"""
Quick fix script for your bee-monitoring-ml setup
"""

import os
import sys
import json

def fix_config():
    """Update config.json to match your actual file structure"""
    
    # Load existing config
    if os.path.exists("config.json"):
        with open("config.json", 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Update paths to match your actual structure
    config["models"] = {
        "binary_classifier": "best_bombus_model.pth",  # Your model is in root
        "object_detector": "models/bombus_object_detection_model.pth", 
        "species_classifier": "models/bombus_species_classifier.pth"
    }
    
    # Save updated config
    with open("config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Updated config.json with correct model paths")

def test_video_analyzer():
    """Test importing your VideoAnalyzer directly from root directory"""
    
    try:
        # Try importing from root directory
        sys.path.insert(0, '.')
        from correct_video_analyzer import VideoAnalyzer
        print("✅ Successfully imported VideoAnalyzer from root directory")
        
        # Test loading your model
        if os.path.exists("best_bombus_model.pth"):
            try:
                analyzer = VideoAnalyzer("best_bombus_model.pth")
                print("✅ Successfully loaded your bombus model")
                return analyzer
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
                return None
        else:
            print("❌ Model file best_bombus_model.pth not found")
            return None
            
    except ImportError as e:
        print(f"❌ Could not import VideoAnalyzer: {e}")
        return None

def find_video_files():
    """Find video files to test with"""
    
    # Check common video locations
    possible_paths = [
        ".",  # Current directory
        "data/",
        "videos/",
        "raw_videos/"
    ]
    
    video_files = []
    
    for path in possible_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.mp4') or f.endswith('.MP4')]
            if files:
                video_files.extend([os.path.join(path, f) for f in files])
    
    if video_files:
        print(f"📹 Found {len(video_files)} video files:")
        for i, video in enumerate(video_files[:5]):  # Show first 5
            print(f"   {i+1}. {video}")
        if len(video_files) > 5:
            print(f"   ... and {len(video_files) - 5} more")
    else:
        print("❌ No video files found in common locations")
        print("💡 Your videos might be on the external drive mentioned in your notes")
    
    return video_files

def test_analysis_pipeline(video_path=None):
    """Test your current analysis pipeline"""
    
    analyzer = test_video_analyzer()
    if not analyzer:
        return None
    
    if not video_path:
        video_files = find_video_files()
        if not video_files:
            print("⚠️ No video files found for testing")
            return None
        video_path = video_files[0]  # Use first found video
    
    print(f"\n🎬 Testing analysis on: {os.path.basename(video_path)}")
    
    try:
        # This should work like your test in the training notes
        results = analyzer.analyze_video(video_path)
        
        print("✅ Analysis completed successfully!")
        print(f"📊 Results summary:")
        if hasattr(results, 'keys'):
            for key, value in results.items():
                if isinstance(value, (int, float, str)):
                    print(f"   {key}: {value}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

def create_fixed_integration():
    """Create a fixed version of the integration script"""
    
    fixed_script = '''
# Fixed import for your setup
import sys
import os
sys.path.insert(0, '.')  # Add root directory to path

try:
    from correct_video_analyzer import VideoAnalyzer
    print("✅ VideoAnalyzer imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    VideoAnalyzer = None

# Modified WorkflowUpgrader class
class FixedWorkflowUpgrader:
    def __init__(self):
        self.model_path = "best_bombus_model.pth"
        
    def test_current_model(self, video_path):
        """Test your current model on a video"""
        if not VideoAnalyzer:
            print("❌ VideoAnalyzer not available")
            return None
            
        try:
            analyzer = VideoAnalyzer(self.model_path)
            results = analyzer.analyze_video(video_path)
            print("✅ Current model test successful")
            return results
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            return None
    
    def simulate_5min_segments(self, video_path):
        """Simulate breaking video into 5-minute segments"""
        results = self.test_current_model(video_path)
        if not results:
            return None
        
        # Extract detection times from your results format
        detection_times = results.get('detection_times', [])
        
        # Group into 5-minute segments
        segments = []
        segment_duration = 300  # 5 minutes
        
        for i, time_str in enumerate(detection_times):
            time_seconds = float(time_str.replace('s', ''))
            segment_num = int(time_seconds // segment_duration)
            
            # Ensure we have enough segments
            while len(segments) <= segment_num:
                segments.append({
                    'segment_number': len(segments) + 1,
                    'start_time': len(segments) * segment_duration,
                    'end_time': (len(segments) + 1) * segment_duration,
                    'detections': [],
                    'detection_count': 0
                })
            
            segments[segment_num]['detections'].append(time_str)
            segments[segment_num]['detection_count'] += 1
        
        print(f"📊 Video broken into {len(segments)} segments:")
        for segment in segments:
            if segment['detection_count'] > 0:
                print(f"   Segment {segment['segment_number']}: {segment['detection_count']} detections")
        
        return segments

if __name__ == "__main__":
    upgrader = FixedWorkflowUpgrader()
    
    # Test with a video file if available
    video_files = []
    for f in os.listdir('.'):
        if f.endswith('.mp4') or f.endswith('.MP4'):
            video_files.append(f)
    
    if video_files:
        test_video = video_files[0]
        print(f"🎬 Testing with: {test_video}")
        segments = upgrader.simulate_5min_segments(test_video)
    else:
        print("💡 No video files found in current directory")
        print("   Try: python fixed_integration.py /path/to/your/video.mp4")
'''
    
    with open("fixed_integration.py", 'w') as f:
        f.write(fixed_script)
    
    print("✅ Created fixed_integration.py")

def main():
    """Run all fixes"""
    print("🔧 FIXING YOUR BEE-MONITORING-ML SETUP")
    print("="*50)
    
    print("\n1. Fixing configuration...")
    fix_config()
    
    print("\n2. Testing VideoAnalyzer import...")
    test_video_analyzer()
    
    print("\n3. Looking for video files...")
    find_video_files()
    
    print("\n4. Creating fixed integration script...")
    create_fixed_integration()
    
    print("\n✅ FIXES COMPLETE!")
    print("\nNext steps:")
    print("1. Test your model: python fixed_integration.py")
    print("2. If you have videos on external drive, mount it and provide path")
    print("3. Run: python integration_script.py --mode upgrade --video /path/to/your/video.mp4")

if __name__ == "__main__":
    main()