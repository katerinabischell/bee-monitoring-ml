#!/usr/bin/env python3
"""
Test the trained YOLO model on specific frames and videos
"""

import os
import cv2
from pathlib import Path

def test_on_annotated_frames():
    """
    Test the model on frames where you manually annotated bees
    """
    
    model_path = "models/object_detection/ventura_bombus_yolo/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🧪 TESTING TRAINED YOLO MODEL")
    print("="*50)
    print(f"Model: {model_path}")
    
    try:
        from ultralytics import YOLO
        
        # Load your trained model
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
        
        # Test video
        video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milkvetch/week 5/day 1/site 1/afternoon/P1000446.MP4"
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return
        
        # Test specific frames where you manually annotated bees
        test_cases = [
            {'timestamp': 2, 'expected': 1, 'note': '1st bombus on the left'},
            {'timestamp': 15, 'expected': 2, 'note': '1st and 2nd bombus visible both flying'},
            {'timestamp': 242, 'expected': 1, 'note': '4th bombus very visible bottom right'},
            {'timestamp': 326, 'expected': 2, 'note': '6th and 7th bombus clearly visible - TWO BEES!'},
            {'timestamp': 336, 'expected': 2, 'note': '6th and 7th bombus clearly visible - TWO BEES!'},
            {'timestamp': 355, 'expected': 2, 'note': '8th bombus + 6th bombus - TWO BEES!'},
            {'timestamp': 364, 'expected': 1, 'note': '8th bombus clearly visible on left'},
            {'timestamp': 386, 'expected': 1, 'note': '8th bombus top left'},
        ]
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n📊 TESTING ON MANUALLY ANNOTATED FRAMES:")
        print(f"{'Time':>4s} | {'Expected':>8s} | {'Detected':>8s} | {'Match':>5s} | {'Confidence':>10s} | Note")
        print("-" * 80)
        
        correct_detections = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            timestamp = test_case['timestamp']
            expected = test_case['expected']
            note = test_case['note']
            
            # Jump to specific frame
            frame_num = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Run detection
                results = model(frame, conf=0.3, verbose=False)
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                
                # Get confidence scores
                confidences = results[0].boxes.conf.tolist() if results[0].boxes is not None else []
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Check if detection matches expectation
                match = "✅" if detections == expected else "❌"
                if detections == expected:
                    correct_detections += 1
                
                print(f"{timestamp:4d}s | {expected:8d} | {detections:8d} | {match:>5s} | {avg_conf:10.3f} | {note[:30]}")
            else:
                print(f"{timestamp:4d}s | {expected:8d} | ERROR    | ❌    | N/A        | Could not read frame")
        
        cap.release()
        
        accuracy = correct_detections / total_tests
        
        print("-" * 80)
        print(f"📈 ACCURACY: {correct_detections}/{total_tests} = {accuracy:.1%}")
        
        if accuracy >= 0.75:
            print("🎉 Excellent! Model performs well on annotated frames")
        elif accuracy >= 0.5:
            print("👍 Good performance, could be improved with more training")
        else:
            print("⚠️ Model needs improvement - consider more training data")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def test_on_video_segment():
    """
    Test the model on a 30-second video segment to see bee detection over time
    """
    
    model_path = "models/object_detection/ventura_bombus_yolo/weights/best.pt"
    video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections/ventura_milkvetch/week 5/day 1/site 1/afternoon/P1000446.MP4"
    
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("❌ Model or video not found")
        return
    
    try:
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n🎬 TESTING ON VIDEO SEGMENT (320-350s - peak activity period)")
        print("="*60)
        
        # Test 30-second segment during peak activity
        start_time = 320  # Start of peak activity
        end_time = 350    # 30 seconds
        
        detections_over_time = []
        
        for timestamp in range(start_time, end_time + 1, 2):  # Every 2 seconds
            frame_num = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                results = model(frame, conf=0.3, verbose=False)
                bee_count = len(results[0].boxes) if results[0].boxes is not None else 0
                detections_over_time.append((timestamp, bee_count))
                
                if bee_count > 0:
                    print(f"   {timestamp:3d}s: {bee_count} bees detected")
        
        cap.release()
        
        # Summary
        total_detections = sum(count for _, count in detections_over_time)
        frames_with_bees = sum(1 for _, count in detections_over_time if count > 0)
        max_bees = max(count for _, count in detections_over_time)
        
        print(f"\n📊 SEGMENT SUMMARY:")
        print(f"   Time period: {start_time}-{end_time}s")
        print(f"   Total detections: {total_detections}")
        print(f"   Frames with bees: {frames_with_bees}/{len(detections_over_time)}")
        print(f"   Max bees in frame: {max_bees}")
        print(f"   Detection rate: {frames_with_bees/len(detections_over_time):.1%}")
        
    except Exception as e:
        print(f"❌ Error in video segment test: {e}")

def main():
    """
    Run both tests
    """
    
    print("🐝 TESTING VENTURA BOMBUS YOLO MODEL")
    print("="*60)
    
    # Test 1: Specific annotated frames
    test_on_annotated_frames()
    
    # Test 2: Video segment analysis
    test_on_video_segment()
    
    print(f"\n✅ TESTING COMPLETE!")
    print("This model can now be used for:")
    print("• Counting individual bees in video frames")
    print("• Locating bee positions with bounding boxes") 
    print("• Analyzing temporal patterns of bee activity")
    print("• Processing both Ventura Milk Vetch and Birds Beak videos")

if __name__ == "__main__":
    main()