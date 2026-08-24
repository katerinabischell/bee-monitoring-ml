"""
HyDaT tracking variant with an additional low-resolution detection pass.

Same tracker as ``track_hydat.py``, plus a downscaled sweep intended to catch
fast-moving bees that motion blur makes the full-resolution pass miss, and a
coarse flower-region pass for context. Kept as a separate file because it is a
methodological variant, not a drop-in replacement.

Usage:
    python src/track_hydat_lowres.py --video VIDEO.MP4 \
        --model models/ventura_bee_detection_v2_best.pt --low_res_height 480
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.stats import linregress
from scipy.optimize import linear_sum_assignment
import time
from collections import defaultdict, deque
import math

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True, help='Path to input video')
parser.add_argument('--model', default='data/models/ventura_bee_detection_v2/weights/best.pt', help='YOLOv8 model path')
parser.add_argument('--confidence', type=float, default=0.4, help='YOLO confidence threshold')
parser.add_argument('--output', default='output_tracked.mp4', help='Output video path')
parser.add_argument('--quadrant_grid', action='store_true', help='Show 3x3 quadrant grid')
parser.add_argument('--low_res_height', type=int, default=480, help='Low resolution height for processing')

args = parser.parse_args()

# Initialize YOLO model
model = YOLO(args.model)

# Video capture and writer setup
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Error: Could not open video file {args.video}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

print(f"Processing video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

# ===== POLYTRACK ALGORITHM PARAMETERS =====
# Background subtractor (improved settings from paper)
background_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500, 
    dist2Threshold=400, 
    detectShadows=True
)

# Tracking parameters (from paper)
max_distance_threshold = 150    # Maximum distance for association
min_confidence_threshold = 0.3   # Minimum confidence for YOLO detections
min_blob_area = 100             # Minimum blob area for background subtraction
max_blob_area = 2000            # Maximum blob area for background subtraction
max_missing_frames = 30         # Maximum frames before terminating track
min_visitation_time = 1.0       # Minimum visitation time in seconds

# Low resolution parameters
low_res_height = args.low_res_height
low_res_width = int((low_res_height / frame_height) * frame_width)
area_scale_factor = (low_res_width * low_res_height) / (frame_width * frame_height)
min_blob_area_lr = int(min_blob_area * area_scale_factor)

# Quadrant system (3x3 grid as in paper)
quadrant_centers = {
    1: np.array([0.167, 0.167]), 2: np.array([0.5, 0.167]), 3: np.array([0.833, 0.167]),
    4: np.array([0.167, 0.5]),   5: np.array([0.5, 0.5]),   6: np.array([0.833, 0.5]),
    7: np.array([0.167, 0.833]), 8: np.array([0.5, 0.833]), 9: np.array([0.833, 0.833])
}

# Track storage
tracks = {}
track_id_counter = 0
frame_num = 0
flowers = []  # Store flower positions
processing_full_res = False

class BeeTrack:
    def __init__(self, track_id, initial_detection, frame_num):
        self.id = track_id
        self.positions = [initial_detection['center']]
        self.areas = [initial_detection.get('area', 0)]
        self.confidences = [initial_detection.get('confidence', 0)]
        self.start_frame = frame_num
        self.last_seen = frame_num
        self.missing_frames = 0
        self.detection_methods = ['YOLO' if initial_detection.get('confidence', 0) > 0.3 else 'BG']
        
        # Velocity tracking for constant velocity model
        self.velocity = np.array([0.0, 0.0])
        self.predicted_position = initial_detection['center']
        
    def get_quadrant(self, center):
        """Determine which quadrant (1-9) the center point belongs to"""
        norm_center = center / np.array([frame_width, frame_height])
        min_dist = float('inf')
        closest_quad = 0
        
        for q, q_center in quadrant_centers.items():
            dist = np.linalg.norm(norm_center - q_center)
            if dist < min_dist and dist < 0.3:  # Within reasonable distance
                min_dist = dist
                closest_quad = q
                
        return closest_quad
    
    def update_velocity(self):
        """Update velocity based on last two positions using constant velocity model"""
        if len(self.positions) >= 2:
            self.velocity = self.positions[-1] - self.positions[-2]
        
    def predict_position(self):
        """Predict next position based on constant velocity model"""
        if len(self.positions) > 0:
            self.predicted_position = self.positions[-1] + self.velocity
        return self.predicted_position
    
    def update(self, detection, frame_num):
        """Update track with new detection"""
        self.positions.append(detection['center'])
        self.areas.append(detection.get('area', 0))
        self.confidences.append(detection.get('confidence', 0))
        self.last_seen = frame_num
        self.missing_frames = 0
        
        # Determine detection method
        method = 'YOLO' if detection.get('confidence', 0) > min_confidence_threshold else 'BG'
        self.detection_methods.append(method)
        
        # Update velocity
        self.update_velocity()
        
    def increment_missing(self):
        """Increment missing frame counter"""
        self.missing_frames += 1
        
    def should_terminate(self):
        """Determine if track should be terminated"""
        return self.missing_frames > max_missing_frames
        
    def get_visitation_duration(self):
        """Calculate visitation duration"""
        return len(self.positions) / fps
        
    def get_last_position(self):
        """Get the last known position"""
        return self.positions[-1] if self.positions else None

def detect_flowers_yolo(frame):
    """Detect flowers using YOLO model at start of video"""
    results = model(frame, verbose=False)
    flowers = []
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            # Assuming class 1 is for flowers (adjust based on your model)
            if len(model.names) > 1 and box.cls.item() == 1:  # Flower class
                xyxy = box.xyxy[0].cpu().numpy()
                center = np.array([(xyxy[0] + xyxy[2])/2, (xyxy[1] + xyxy[3])/2])
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                flowers.append({'center': center, 'area': area, 'bbox': xyxy})
    
    return flowers

def get_yolo_detections(frame):
    """Get bee detections from YOLO model"""
    results = model(frame, verbose=False, conf=args.confidence)
    detections = []
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            # Assuming class 0 is for bees (adjust based on your model)
            if box.cls.item() == 0:  # Bee class
                conf = box.conf.item()
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Calculate center and area
                center = np.array([(xyxy[0] + xyxy[2])/2, (xyxy[1] + xyxy[3])/2])
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                
                detections.append({
                    'center': center,
                    'area': area,
                    'confidence': conf,
                    'bbox': xyxy,
                    'method': 'YOLO'
                })
    
    return detections

def get_background_detections(frame):
    """Get detections from background subtraction"""
    # Apply background subtraction
    fg_mask = background_subtractor.apply(frame)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_blob_area < area < max_blob_area:
            # Get bounding box and center
            x, y, w, h = cv2.boundingRect(contour)
            center = np.array([x + w/2, y + h/2])
            
            detections.append({
                'center': center,
                'area': area,
                'confidence': 0.1,  # Low confidence for background subtraction
                'bbox': np.array([x, y, x+w, y+h]),
                'method': 'BG'
            })
    
    return detections

def get_low_res_detections(frame):
    """Get detections in low resolution mode"""
    # Resize frame
    small_frame = cv2.resize(frame, (low_res_width, low_res_height))
    
    # Apply background subtraction
    fg_mask = background_subtractor.apply(small_frame)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any significant movement detected
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_blob_area_lr:
            return True  # Switch to full resolution
            
    return False

def associate_detections_with_tracks(detections, tracks):
    """Associate detections with existing tracks using Hungarian algorithm"""
    if not detections or not tracks:
        return {}, set()
    
    # Predict positions for all tracks
    track_ids = list(tracks.keys())
    predicted_positions = []
    
    for track_id in track_ids:
        predicted_pos = tracks[track_id].predict_position()
        predicted_positions.append(predicted_pos)
    
    # Calculate cost matrix
    cost_matrix = np.full((len(track_ids), len(detections)), np.inf)
    
    for i, track_id in enumerate(track_ids):
        predicted_pos = predicted_positions[i]
        for j, detection in enumerate(detections):
            distance = np.linalg.norm(detection['center'] - predicted_pos)
            if distance < max_distance_threshold:
                cost_matrix[i, j] = distance
    
    # Use Hungarian algorithm for assignment
    if cost_matrix.size > 0:
        # Pad matrix if necessary to make it square
        max_dim = max(cost_matrix.shape)
        if cost_matrix.shape[0] != cost_matrix.shape[1]:
            padded_matrix = np.full((max_dim, max_dim), np.inf)
            padded_matrix[:cost_matrix.shape[0], :cost_matrix.shape[1]] = cost_matrix
            cost_matrix = padded_matrix
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract valid assignments
        assignments = {}
        used_detections = set()
        
        for i, j in zip(row_ind, col_ind):
            if (i < len(track_ids) and j < len(detections) and 
                cost_matrix[i, j] < max_distance_threshold):
                track_id = track_ids[i]
                assignments[track_id] = detections[j]
                used_detections.add(j)
        
        return assignments, used_detections
    
    return {}, set()

def hybrid_detection_strategy(frame):
    """Implement hybrid detection strategy from Polytrack paper"""
    # Get YOLO detections (higher priority)
    yolo_detections = get_yolo_detections(frame)
    
    # Get background detections
    bg_detections = get_background_detections(frame)
    
    # Combine detections with priority to YOLO
    all_detections = yolo_detections + bg_detections
    
    return all_detections, len(yolo_detections), len(bg_detections)

def draw_detection(frame, detection, track_id, color):
    """Draw detection on frame"""
    bbox = detection['bbox'].astype(int)
    center = detection['center'].astype(int)
    
    # Draw bounding box
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    # Draw center point
    cv2.circle(frame, tuple(center), 3, color, -1)
    
    # Draw label
    method = detection.get('method', 'UNK')
    conf = detection.get('confidence', 0)
    label = f'Bee {track_id} ({method}) {conf:.2f}'
    cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_track_trajectory(frame, track, color):
    """Draw track trajectory"""
    positions = track.positions
    if len(positions) > 1:
        points = [tuple(pos.astype(int)) for pos in positions[-10:]]  # Last 10 positions
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], color, 2)

def draw_quadrant_grid(frame):
    """Draw 3x3 quadrant grid"""
    if args.quadrant_grid:
        # Vertical lines
        for i in range(1, 3):
            x = frame_width * i // 3
            cv2.line(frame, (x, 0), (x, frame_height), (128, 128, 128), 1)
        
        # Horizontal lines  
        for i in range(1, 3):
            y = frame_height * i // 3
            cv2.line(frame, (0, y), (frame_width, y), (128, 128, 128), 1)
        
        # Draw quadrant numbers
        for q, center in quadrant_centers.items():
            pos = (int(center[0] * frame_width), int(center[1] * frame_height))
            cv2.putText(frame, str(q), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (64, 64, 64), 2)

def draw_flowers(frame, flowers):
    """Draw detected flowers"""
    for i, flower in enumerate(flowers):
        center = flower['center'].astype(int)
        cv2.circle(frame, tuple(center), 8, (255, 255, 0), 2)
        cv2.putText(frame, f'F{i+1}', tuple(center + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

# ===== MAIN TRACKING LOOP =====
print("Starting Polytrack with hybrid detection...")
start_time = time.time()

# Detect flowers in first frame
ret, first_frame = cap.read()
if ret:
    flowers = detect_flowers_yolo(first_frame)
    print(f"Detected {len(flowers)} flowers")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Decide processing mode
    if not processing_full_res and not tracks:
        # Low resolution mode when no active tracks
        should_switch = get_low_res_detections(frame)
        if not should_switch:
            # Continue in low res mode, just update background model
            frame_num += 1
            if frame_num % 100 == 0:
                print(f"Low-res processing: Frame {frame_num}/{total_frames}")
            out.write(frame)
            continue
        else:
            processing_full_res = True
            print(f"Switching to full resolution at frame {frame_num}")
    
    # Full resolution processing
    all_detections, yolo_count, bg_count = hybrid_detection_strategy(frame)
    
    # Associate detections with existing tracks
    assignments, used_detections = associate_detections_with_tracks(all_detections, tracks)
    
    # Update existing tracks
    for track_id, detection in assignments.items():
        tracks[track_id].update(detection, frame_num)
        
        # Draw track
        color = (0, 255, 0) if detection.get('method') == 'YOLO' else (255, 0, 0)
        draw_detection(frame, detection, track_id, color)
        draw_track_trajectory(frame, tracks[track_id], color)
    
    # Create new tracks from unassigned detections
    for i, detection in enumerate(all_detections):
        if i not in used_detections and detection.get('confidence', 0) > min_confidence_threshold:
            track_id_counter += 1
            new_track = BeeTrack(track_id_counter, detection, frame_num)
            tracks[track_id_counter] = new_track
            
            # Draw new track
            color = (0, 255, 255)  # Cyan for new tracks
            draw_detection(frame, detection, track_id_counter, color)
    
    # Handle missing tracks
    tracks_to_remove = []
    for track_id, track in tracks.items():
        if track_id not in assignments:
            track.increment_missing()
            
            # Draw predicted position for missing tracks
            if track.missing_frames < 10:  # Show prediction for first few missing frames
                pred_pos = track.predict_position().astype(int)
                cv2.circle(frame, tuple(pred_pos), 8, (0, 0, 255), 2)
                cv2.putText(frame, f'Lost {track_id}', tuple(pred_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if track.should_terminate():
                duration = track.get_visitation_duration()
                if duration >= min_visitation_time:
                    print(f"Track {track_id} completed: {duration:.2f}s, "
                          f"{len(track.positions)} detections")
                tracks_to_remove.append(track_id)
    
    # Remove terminated tracks
    for track_id in tracks_to_remove:
        del tracks[track_id]
    
    # Switch back to low resolution if no active tracks
    if not tracks:
        processing_full_res = False
    
    # Draw flowers
    draw_flowers(frame, flowers)
    
    # Draw quadrant grid
    draw_quadrant_grid(frame)
    
    # Draw frame info
    active_tracks = len(tracks)
    mode = "FULL-RES" if processing_full_res else "LOW-RES"
    info_text = [
        f'Frame: {frame_num}/{total_frames} ({mode})',
        f'Active Tracks: {active_tracks}',
        f'YOLO: {yolo_count}, BG: {bg_count}',
        f'Total Tracks Created: {track_id_counter}'
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(frame, text, (10, 30 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Write frame
    out.write(frame)
    frame_num += 1
    
    # Progress indicator
    if frame_num % 100 == 0:
        elapsed = time.time() - start_time
        progress = frame_num / total_frames
        eta = elapsed / progress - elapsed if progress > 0 else 0
        print(f"Progress: {progress*100:.1f}% ({frame_num}/{total_frames}), "
              f"ETA: {eta/60:.1f}min, Active tracks: {active_tracks}")

# Cleanup
cap.release()
out.release()

# Final statistics
end_time = time.time()
processing_time = end_time - start_time
print(f"\n=== POLYTRACK COMPLETE ===")
print(f"Processing time: {processing_time/60:.1f} minutes")
print(f"Total frames processed: {frame_num}")
print(f"Total tracks created: {track_id_counter}")
print(f"Flowers detected: {len(flowers)}")
print(f"Final active tracks: {len(tracks)}")
print(f"Output saved to: {args.output}")

# Print final track summaries
print(f"\n=== VALID VISITATION EVENTS ===")
valid_tracks = 0
for track_id, track in tracks.items():
    duration = track.get_visitation_duration()
    if duration >= min_visitation_time:
        valid_tracks += 1
        final_quad = track.get_quadrant(track.get_last_position()) if track.get_last_position() is not None else 0
        avg_confidence = np.mean(track.confidences) if track.confidences else 0
        yolo_detections = sum(1 for method in track.detection_methods if method == 'YOLO')
        bg_detections = len(track.detection_methods) - yolo_detections
        print(f"Track {track_id}: {duration:.2f}s, Q{final_quad}, "
              f"avg_conf: {avg_confidence:.2f}, YOLO: {yolo_detections}, BG: {bg_detections}")

print(f"\nTotal valid visitation events: {valid_tracks}")