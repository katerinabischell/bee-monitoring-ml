"""
HyDaT hybrid detection-and-tracking over field video (downstream of detection).

Couples the YOLOv8 detector with a background-subtraction fallback and links
detections into tracks across frames, so that a bee is followed through the
frame rather than counted independently in each one. Detections are associated
to tracks by Hungarian assignment; short gaps are bridged so a brief miss does
not split one visit into two.

This is the stage that turns per-frame detections into visitation events. It is
adapted for this dataset from the HyDaT approach of Ratnayake et al.

Usage:
    python src/track_hydat.py --video VIDEO.MP4 \
        --model models/ventura_bee_detection_v2_best.pt --confidence 0.4

See also ``track_hydat_lowres.py``, which adds a downscaled detection pass.
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.stats import linregress
from scipy.optimize import linear_sum_assignment
import time
from collections import defaultdict, deque

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True, help='Path to input video')
parser.add_argument('--model', default='data/models/ventura_bee_detection_v2/weights/best.pt', help='YOLOv8 model path')
parser.add_argument('--confidence', type=float, default=0.4, help='YOLO confidence threshold')
parser.add_argument('--output', default='output_tracked.mp4', help='Output video path')
parser.add_argument('--quadrant_grid', action='store_true', help='Show 3x3 quadrant grid')
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

# ===== POLYTRACK PARAMETERS (ADAPTED FROM PAPER, TUNED FOR BUMBLE BEES) =====
# Background subtractor
background_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400,
    detectShadows=True
)

# Tracking parameters (adjusted for bumble bees: larger size, potentially slower movement)
theta_t = 15  # Threshold for consecutive undetected frames before exit prob calculation
beta_thresh = 0.85  # Exit probability threshold
mdt_threshold_yolo = 200  # Increased max distance threshold for YOLO associations (larger bees)
mdt_threshold_bs = 100  # Increased max distance threshold for background subtraction
min_visitation_time = 5  # Minimum visitation time in seconds (based on typical bumble bee flower visit ~5-7s)
obscurity_thresh = 0.1  # Maximum allowed occlusion ratio
max_lost_frames = 90  # Increased maximum frames before terminating track (allow longer occlusions)
min_blob_area = 200  # Increased minimum blob area for background subtraction (bumble bees are larger)
max_blob_area = 4000  # Increased maximum blob area for background subtraction
low_res_scale = 0.5  # Scale factor for low-resolution mode (e.g., half resolution)

# Quadrant system (optional grid for visualization, no longer used for visits)
quadrant_centers = {
    1: np.array([0.167, 0.167]), 2: np.array([0.5, 0.167]), 3: np.array([0.833, 0.167]),
    4: np.array([0.167, 0.5]), 5: np.array([0.5, 0.5]), 6: np.array([0.833, 0.5]),
    7: np.array([0.167, 0.833]), 8: np.array([0.5, 0.833]), 9: np.array([0.833, 0.833])
}

# Track storage
tracks = {}
track_id_counter = 0
frame_num = 0

class BumbleBeeTrack:
    def __init__(self, track_id, initial_center, initial_area, frame_num):
        self.id = track_id
        self.positions = deque(maxlen=100)  # Store recent positions for prediction
        self.positions.append((initial_center, self.get_quadrant(initial_center)))
        self.areas = deque(maxlen=10)  # Recent areas for occlusion check
        self.areas.append(initial_area)
        self.start_frame = frame_num
        self.last_seen = frame_num
        self.last_pos = initial_center
        self.predicted_pos = initial_center
        self.velocity = np.array([0.0, 0.0])  # For constant velocity model
        self.speeds = deque(maxlen=10)  # Recent speeds
        self.occluded_frames = 0
        self.detection_method = []  # Track detection methods
        self.visitation_time = 0.0  # Cumulative visitation time (updated per frame)

    def get_quadrant(self, center):
        # Optional: kept for grid visualization if enabled
        norm_center = center / np.array([frame_width, frame_height])
        min_dist = float('inf')
        closest_quad = 0
        for q, q_center in quadrant_centers.items():
            dist = np.linalg.norm(norm_center - q_center)
            if dist < min_dist and dist < 0.3:
                min_dist = dist
                closest_quad = q
        return closest_quad

    def update(self, new_center, new_area, frame_num, detection_method):
        # Update velocity and prediction
        if len(self.positions) >= 1:
            prev_center = self.positions[-1][0]
            self.velocity = new_center - prev_center
            distance = np.linalg.norm(self.velocity)
            speed = distance * fps
            self.speeds.append(speed)

        quadrant = self.get_quadrant(new_center)  # Optional for viz
        self.positions.append((new_center, quadrant))
        self.areas.append(new_area)
        self.last_pos = new_center
        self.last_seen = frame_num
        self.detection_method.append(detection_method)
        self.predicted_pos = new_center + self.velocity  # Simple constant velocity prediction
        self.visitation_time += 1 / fps  # Accumulate time spent in frame (visitation)

        # Occlusion check using linear regression on areas
        if len(self.areas) >= 5:
            recent_areas = list(self.areas)[-5:]
            frames = np.arange(len(recent_areas))
            slope, _, _, _, _ = linregress(frames, recent_areas)
            if slope < -10:
                self.occluded_frames += 1
                self.visitation_time -= 1 / fps  # Subtract occluded frame from visitation time

    def predict_next_pos(self):
        # Constant velocity model from paper: Pk = 2*Dk-1 - Dk-2
        if len(self.positions) >= 2:
            dk1 = self.positions[-1][0]
            dk2 = self.positions[-2][0]
            return 2 * dk1 - dk2
        elif len(self.positions) == 1:
            return self.positions[-1][0]
        return self.last_pos

    def calculate_exit_probability(self, lost_frames):
        if lost_frames <= theta_t or len(self.speeds) == 0:
            return 0.0
        dist_to_edge = min(
            self.last_pos[0], frame_width - self.last_pos[0],
            self.last_pos[1], frame_height - self.last_pos[1]
        )
        speeds_array = np.array(list(self.speeds))
        quantiles = [0.25, 0.5, 0.75, 0.9]
        beta = 0
        for q in quantiles:
            speed_quantile = np.quantile(speeds_array, q)
            predicted_distance = speed_quantile * lost_frames / fps
            if predicted_distance >= dist_to_edge:
                beta += (1 - q)
        beta /= len(quantiles)
        return beta

    def should_terminate(self, current_frame):
        lost_frames = current_frame - self.last_seen
        if lost_frames > max_lost_frames:
            return True
        if lost_frames > theta_t:
            exit_prob = self.calculate_exit_probability(lost_frames)
            if exit_prob > beta_thresh:
                return True
        return False

    def get_visitation_duration(self):
        return self.visitation_time

    def get_obscurity_ratio(self):
        return self.occluded_frames / len(self.positions) if self.positions else 0

def get_yolo_detections(frame):
    results = model.track(frame, persist=True, conf=args.confidence, verbose=False)
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            conf = box.conf.item()
            xyxy = box.xyxy[0].cpu().numpy()
            center = np.array([(xyxy[0] + xyxy[2])/2, (xyxy[1] + xyxy[3])/2])
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            track_id = box.id.item() if box.id is not None else None
            detections.append({
                'center': center,
                'area': area,
                'confidence': conf,
                'bbox': xyxy,
                'yolo_track_id': track_id
            })
    return detections

def get_background_detections(frame):
    fg_mask = background_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_blob_area < area < max_blob_area:
            x, y, w, h = cv2.boundingRect(contour)
            center = np.array([x + w/2, y + h/2])
            detections.append({
                'center': center,
                'area': area,
                'confidence': 0.1,
                'bbox': np.array([x, y, x+w, y+h]),
                'yolo_track_id': None
            })
    return detections

def hybrid_detection_strategy(frame, active_tracks, low_res_mode):
    if low_res_mode:
        # Downscale frame
        low_frame = cv2.resize(frame, None, fx=low_res_scale, fy=low_res_scale)
        bg_detections = get_background_detections(low_frame)
        # Scale back detections
        for det in bg_detections:
            det['center'] /= low_res_scale
            det['bbox'] /= low_res_scale
            det['area'] /= (low_res_scale ** 2)
        # If significant blobs, switch to full res YOLO
        if any(d['area'] > min_blob_area / (low_res_scale ** 2) for d in bg_detections):
            yolo_detections = get_yolo_detections(frame)
            return yolo_detections, [], {'yolo': len(yolo_detections), 'background': 0}
        return [], bg_detections, {'yolo': 0, 'background': len(bg_detections)}
    else:
        # Full res: use BG if low changes, else YOLO
        bg_detections = get_background_detections(frame)
        if len(bg_detections) <= len(active_tracks) + 1:  # Low changes
            return [], bg_detections, {'yolo': 0, 'background': len(bg_detections)}
        else:
            yolo_detections = get_yolo_detections(frame)
            return yolo_detections, [], {'yolo': len(yolo_detections), 'background': 0}

def associate_detections(predictions, detections, max_dist):
    if not detections:
        return {}
    cost_matrix = np.zeros((len(predictions), len(detections)))
    for i, pred in enumerate(predictions):
        for j, det in enumerate(detections):
            dist = np.linalg.norm(pred - det['center'])
            cost_matrix[i, j] = dist if dist < max_dist else np.inf
    # Pad to square if needed
    n_rows, n_cols = cost_matrix.shape
    if n_rows > n_cols:
        cost_matrix = np.pad(cost_matrix, ((0, 0), (0, n_rows - n_cols)), constant_values=np.inf)
    elif n_cols > n_rows:
        cost_matrix = np.pad(cost_matrix, ((0, n_cols - n_rows), (0, 0)), constant_values=np.inf)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    associations = {}
    for r, c in zip(row_ind, col_ind):
        if c < len(detections) and cost_matrix[r, c] < np.inf:
            associations[r] = detections[c]
    return associations

def draw_detection(frame, detection, track_id, color, method_label):
    bbox = detection['bbox'].astype(int)
    center = detection['center'].astype(int)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.circle(frame, tuple(center), 3, color, -1)
    label = f'BumbleBee {track_id} ({method_label}) {detection["confidence"]:.2f}'
    cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_quadrant_grid(frame):
    if args.quadrant_grid:
        for i in range(1, 3):
            x = frame_width * i // 3
            cv2.line(frame, (x, 0), (x, frame_height), (128, 128, 128), 1)
        for i in range(1, 3):
            y = frame_height * i // 3
            cv2.line(frame, (0, y), (frame_width, y), (128, 128, 128), 1)
        for q, center in quadrant_centers.items():
            pos = (int(center[0] * frame_width), int(center[1] * frame_height))
            cv2.putText(frame, str(q), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (64, 64, 64), 2)

# ===== MAIN TRACKING LOOP =====
print("Starting Polytrack-inspired hybrid tracking for bumble bees...")
start_time = time.time()

low_res_mode = True
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    active_tracks = {tid: track for tid, track in tracks.items() if frame_num - track.last_seen < theta_t}
    low_res_mode = len(active_tracks) == 0

    # Get detections
    yolo_detections, bg_detections, detection_counts = hybrid_detection_strategy(frame, active_tracks, low_res_mode)
    all_detections = yolo_detections + bg_detections

    # Get predictions for active tracks
    predictions = []
    track_ids = list(active_tracks.keys())
    for tid in track_ids:
        pred_pos = active_tracks[tid].predict_next_pos()
        predictions.append(pred_pos)

    # Associate using Hungarian
    associations = associate_detections(predictions, all_detections, mdt_threshold_yolo if yolo_detections else mdt_threshold_bs)

    # Update associated tracks
    used_detections = set()
    for idx, detection in associations.items():
        tid = track_ids[idx]
        method = 'YOLO' if detection in yolo_detections else 'BG'
        tracks[tid].update(detection['center'], detection['area'], frame_num, method)
        draw_detection(frame, detection, tid, (0, 255, 0) if method == 'YOLO' else (255, 0, 0), method)
        used_detections.add(all_detections.index(detection))

    # Create new tracks from unassociated high-conf detections (prefer YOLO)
    for i, det in enumerate(all_detections):
        if i not in used_detections and det['confidence'] > 0.5:  # Threshold for new tracks
            track_id_counter += 1
            new_track = BumbleBeeTrack(track_id_counter, det['center'], det['area'], frame_num)
            tracks[track_id_counter] = new_track
            draw_detection(frame, det, track_id_counter, (0, 255, 255), 'NEW')

    # Handle lost/missing tracks
    tracks_to_remove = []
    for tid, track in list(tracks.items()):
        if track.should_terminate(frame_num):
            duration = track.get_visitation_duration()
            obscurity = track.get_obscurity_ratio()
            if duration >= min_visitation_time and obscurity <= obscurity_thresh:
                print(f"Track {tid} completed: visitation duration {duration:.2f}s, obscurity: {obscurity*100:.1f}%")
            tracks_to_remove.append(tid)
        elif track.last_seen < frame_num:
            # Search in predicted radius
            pred_pos = track.predict_next_pos()
            for det in all_detections:
                if np.linalg.norm(det['center'] - pred_pos) < mdt_threshold_bs:
                    method = 'YOLO' if det in yolo_detections else 'BG'
                    track.update(det['center'], det['area'], frame_num, method + '-reacq')
                    draw_detection(frame, det, tid, (255, 255, 0), 'REACQ')
                    break
            else:
                # Draw predicted/lost
                if track.predicted_pos is not None:
                    pos = track.predicted_pos.astype(int)
                    cv2.circle(frame, tuple(pos), 8, (0, 0, 255), 2)
                    cv2.putText(frame, f'Lost {tid}', tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for tid in tracks_to_remove:
        del tracks[tid]

    draw_quadrant_grid(frame)

    # Frame info
    active_count = len(active_tracks)
    info_text = [
        f'Frame: {frame_num}/{total_frames}',
        f'Active Tracks: {active_count}',
        f'YOLO: {detection_counts["yolo"]}, BG: {detection_counts["background"]}',
        f'Total Tracks: {track_id_counter}',
        f'Low-Res: {"Yes" if low_res_mode else "No"}'
    ]
    for i, text in enumerate(info_text):
        cv2.putText(frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)
    frame_num += 1

    if frame_num % 100 == 0:
        elapsed = time.time() - start_time
        progress = frame_num / total_frames
        eta = elapsed / progress - elapsed if progress > 0 else 0
        print(f"Progress: {progress*100:.1f}% ({frame_num}/{total_frames}), ETA: {eta/60:.1f}min")

cap.release()
out.release()

end_time = time.time()
processing_time = end_time - start_time
print(f"\n=== TRACKING COMPLETE ===")
print(f"Processing time: {processing_time/60:.1f} minutes")
print(f"Total frames: {frame_num}")
print(f"Total tracks created: {track_id_counter}")
print(f"Output: {args.output}")

print("\n=== VALID BUMBLE BEE VISITATION EVENTS ===")
valid_tracks = 0
total_visitation_time = 0.0
for tid, track in tracks.items():
    duration = track.get_visitation_duration()
    obscurity = track.get_obscurity_ratio()
    if duration >= min_visitation_time and obscurity <= obscurity_thresh:
        valid_tracks += 1
        total_visitation_time += duration
        avg_speed = np.mean(track.speeds) if track.speeds else 0
        print(f"Track {tid}: visitation duration {duration:.2f}s, speed: {avg_speed:.1f}px/s, obscurity: {obscurity*100:.1f}%")
print(f"Total unique bumble bee visits: {valid_tracks}")
print(f"Total visitation time: {total_visitation_time:.2f} seconds")