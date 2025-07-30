
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
