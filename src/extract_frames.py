#!/usr/bin/env python3
"""
Extract still frames from field video and scaffold YOLO labels (pipeline stage 1 of 6).

Samples frames at fixed intervals with OpenCV and seeds a label file for each
frame from the hand-kept observation log, which records, per timestamp, whether
a bee was present and which cell of a 3x3 quadrant grid it occupied. The boxes
this produces are grid-centre placeholders: they mark *which frames need
attention*, and are replaced with real, tightly drawn boxes in ``annotate.py``.

Placeholder boxes must never be trained on directly. The published model was
trained only on frames that went through the annotation stage.

Raw video is not distributed with this repository (see data/README.md); set
``NCOS_VIDEO_ROOT`` to a local copy of the season archive.

Usage:
    NCOS_VIDEO_ROOT=/path/to/videos python src/extract_frames.py

Next stage: ``annotate.py``.
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

_VIDEO_ROOT = os.environ.get("NCOS_VIDEO_ROOT", "videos")
# Raw field video is not distributed with this repository (see data/README.md).
# Point NCOS_VIDEO_ROOT at your local copy of the season archive, or pass --video-root.

class VenturaDataset:
    """
    Create YOLO dataset from Ventura Milk Vetch videos with quadrant-based bee tracking

    Quadrant system:
    1 = top left 2 = top middle 3 = top right
    4 = middle left 5 = middle middle 6 = middle right
    7 = bottom left 8 = bottom middle 9 = bottom right
    """
    def __init__(self, output_dir="data", video_id="P1000471"):
        self.video_id = video_id
        self.base_dir = Path(output_dir)
        self.extracted_frames_dir = self.base_dir / f"extracted_frames_{video_id}"
        self.annotations_dir = self.base_dir / f"annotations_{video_id}"
        self.setup_directory_structure()

        # Define quadrant centers for bounding box placement
        self.quadrant_centers = {
            1: (0.25, 0.25), # top left
            2: (0.50, 0.25), # top middle
            3: (0.75, 0.25), # top right
            4: (0.25, 0.50), # middle left
            5: (0.50, 0.50), # middle middle
            6: (0.75, 0.50), # middle right
            7: (0.25, 0.75), # bottom left
            8: (0.50, 0.75), # bottom middle
            9: (0.75, 0.75), # bottom right
        }

    def setup_directory_structure(self):
        """Create organized directory structure for the full pipeline"""
        dirs = [
            self.base_dir / "raw_videos",
            self.extracted_frames_dir,
            self.annotations_dir,
            self.base_dir / "models",
            self.base_dir / "results",
            self.base_dir / "config"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory structure for {self.video_id} in {self.base_dir}")

    def parse_timestamp_to_seconds(self, timestamp_str):
        """Convert timestamp string (MM:SS) to seconds"""
        try:
            parts = timestamp_str.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        except:
            print(f"⚠️ Could not parse timestamp: {timestamp_str}")
            return None

    def parse_quadrants(self, quadrant_str):
        """Parse quadrant string into list of integers representing frame regions"""
        if not quadrant_str or quadrant_str.strip() == "":
            return []
        # Handle cases like "5, 5" or "7, 5, 3"
        quadrants = []
        for quad in quadrant_str.split(','):
            quad = quad.strip()
            if quad.isdigit() and 1 <= int(quad) <= 9:
                quadrants.append(int(quad))
        return quadrants

    def get_quadrant_name(self, quadrant_num):
        """Get human-readable name for quadrant"""
        quadrant_names = {
            1: "top-left", 2: "top-middle", 3: "top-right",
            4: "middle-left", 5: "middle-center", 6: "middle-right",
            7: "bottom-left", 8: "bottom-middle", 9: "bottom-right"
        }
        return quadrant_names.get(quadrant_num, f"unknown-{quadrant_num}")

    def prepare_ventura_dataset(self, manual_annotations, video_path=None):
        """
        Prepare dataset using manual annotations for a given video

        Args:
            manual_annotations: List of manual annotations with timestamps and quadrant locations
            video_path: Optional specific path to the video file
        """
        print(f"\n{'='*70}")
        print(f"PREPARING VENTURA MILK VETCH {self.video_id} DATASET")
        print(f"Using Quadrant-Based Bee Location System")
        print(f"{'='*70}")

        # Video paths
        possible_video_paths = [
            f"{_VIDEO_ROOT}/ventura_milkvetch/week 6/day 1/site 1/morning/{self.video_id}.MP4",
            f"{_VIDEO_ROOT}/ventura_milk_vetch/week 6/day 1/site 1/morning/{self.video_id}.MP4",
            f"{_VIDEO_ROOT}/week 6/day 2/site 2/afternoon/{self.video_id}.MP4",
            f"{_VIDEO_ROOT}/{self.video_id}.MP4",
            f"data/raw_videos/{self.video_id}.MP4",
            f"{self.video_id}.MP4"
        ]
        if video_path:
            possible_video_paths.insert(0, video_path)

        video_path = None
        for path in possible_video_paths:
            if os.path.exists(path):
                video_path = path
                break

        if not video_path:
            print(f"❌ Video {self.video_id}.MP4 not found in expected locations:")
            for path in possible_video_paths:
                print(f" - {path}")
            print("💡 Please provide the correct path to {self.video_id}.MP4")
            return None, None

        print(f"✅ Found video: {video_path}")

        # Extract frames
        extracted_frames = self.extract_frames_from_manual_annotations(
            video_path, manual_annotations, f"{self.video_id}"
        )

        # Create config
        config_path = self.create_yolo_config()

        # Generate summary - all frames have bees
        total_frames = len(extracted_frames)
        total_bees = sum([len(f['quadrants']) for f in extracted_frames])

        # Quadrant usage statistics
        quadrant_usage = {}
        for frame in extracted_frames:
            for quad in frame['quadrants']:
                quadrant_usage[quad] = quadrant_usage.get(quad, 0) + 1

        print(f"\n📊 VENTURA MILK VETCH {self.video_id} DATASET SUMMARY:")
        print(f" Total frames with bees: {total_frames}")
        print(f" Total bee instances: {total_bees}")
        print(f" Average bees per frame: {total_bees/total_frames:.2f}")
        print(f" Plant type: Ventura Milk Vetch")

        print(f"\n📍 QUADRANT USAGE STATISTICS:")
        for quad in sorted(quadrant_usage.keys()):
            quad_name = self.get_quadrant_name(quad)
            count = quadrant_usage[quad]
            print(f" Quadrant {quad} ({quad_name}): {count} instances")

        return extracted_frames, config_path

    def extract_frames_from_manual_annotations(self, video_path, annotations, output_name_prefix):
        """Extract frames from video at manually annotated timestamps"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        extracted_frames = []

        print(f"🎬 Extracting frames from {Path(video_path).name}")
        print(f" Total annotations: {len(annotations)}")
        print(f" Video FPS: {fps:.2f}")

        for i, annotation in enumerate(annotations):
            timestamp_seconds = annotation['timestamp_seconds']
            quadrants = annotation['quadrants']
            has_bee = len(quadrants) > 0

            # Jump to timestamp
            frame_number = int(timestamp_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Could not extract frame at {timestamp_seconds}s")
                continue

            # Save frame
            frame_filename = f"{output_name_prefix}_t{int(timestamp_seconds):04d}.jpg"
            frame_path = self.extracted_frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            # Create initial YOLO annotation
            label_filename = f"{output_name_prefix}_t{int(timestamp_seconds):04d}.txt"
            label_path = self.annotations_dir / label_filename

            # All timestamps have bees (create bounding boxes based on quadrants)
            bbox_annotations = []

            for quad in quadrants:
                if quad in self.quadrant_centers:
                    # Get quadrant center coordinates
                    x_center, y_center = self.quadrant_centers[quad]

                    # Default bounding box size (will be refined in annotation step)
                    width = 0.20 # 20% of frame width
                    height = 0.20 # 20% of frame height

                    # Create YOLO format annotation (class x_center y_center width height)
                    bbox_annotation = f"0 {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}"
                    bbox_annotations.append(bbox_annotation)

            with open(label_path, 'w') as f:
                for bbox in bbox_annotations:
                    f.write(bbox + "\n")

            quadrants_str = ", ".join([f"{q}({self.get_quadrant_name(q)})" for q in quadrants])
            print(f" ✅ Frame {i+1}: {timestamp_seconds}s -> {len(quadrants)} BEE(S) in quadrants: {quadrants_str}")

            extracted_frames.append({
                'timestamp': annotation['timestamp'],
                'timestamp_seconds': timestamp_seconds,
                'frame_filename': frame_filename,
                'label_filename': label_filename,
                'has_bee': True, # All timestamps have bees
                'quadrants': quadrants,
                'bee_count': len(quadrants),
                'quadrant_names': [self.get_quadrant_name(q) for q in quadrants],
                'plant_type': 'ventura_milk_vetch',
                'video': self.video_id,
                'needs_bbox_refinement': True # All frames need refinement
            })

        cap.release()

        print(f"✅ Extracted {len(extracted_frames)} frames from {Path(video_path).name}")
        return extracted_frames

    def create_yolo_config(self):
        """Create YOLO configuration files"""
        dataset_config = {
            'path': str(self.base_dir.absolute()),
            'train': f'extracted_frames_{self.video_id}',
            'val': f'extracted_frames_{self.video_id}',
            'test': f'extracted_frames_{self.video_id}',
            'nc': 1,
            'names': ['bombus']
        }

        config_path = self.base_dir / "config" / f"dataset_{self.video_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"✅ Created YOLO config: {config_path}")
        return config_path

    def save_annotation_summary(self, extracted_frames):
        """Save a summary of annotations for reference"""
        summary_data = []

        for frame in extracted_frames:
            summary_data.append({
                'timestamp': frame['timestamp'],
                'timestamp_seconds': frame['timestamp_seconds'],
                'frame_filename': frame['frame_filename'],
                'has_bee': frame['has_bee'],
                'bee_count': frame['bee_count'],
                'quadrants': ', '.join(map(str, frame['quadrants'])) if frame['quadrants'] else '',
                'quadrant_names': ', '.join(frame['quadrant_names']) if frame['quadrant_names'] else '',
                'needs_annotation_refinement': frame['needs_bbox_refinement']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.base_dir / f"annotation_summary_{self.video_id}.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"✅ Saved annotation summary: {summary_path}")
        return summary_path

    def create_quadrant_reference(self):
        """Create a visual reference for the quadrant system"""
        reference_content = """
# Ventura Milkvetch Quadrant Reference
## Quadrant System Layout:
```
┌─────────┬─────────┬─────────┐
│ 1 │ 2 │ 3 │
│top-left │top-mid │top-right│
├─────────┼─────────┼─────────┤
│ 4 │ 5 │ 6 │
│mid-left │ center │mid-right│
├─────────┼─────────┼─────────┤
│ 7 │ 8 │ 9 │
│bot-left │bot-mid │bot-right│
└─────────┴─────────┴─────────┘
```
## Coordinates Used for Bounding Boxes:
- Quadrant 1 (top-left): Center at (0.25, 0.25)
- Quadrant 2 (top-middle): Center at (0.50, 0.25)
- Quadrant 3 (top-right): Center at (0.75, 0.25)
- Quadrant 4 (middle-left): Center at (0.25, 0.50)
- Quadrant 5 (middle-center): Center at (0.50, 0.50)
- Quadrant 6 (middle-right): Center at (0.75, 0.50)
- Quadrant 7 (bottom-left): Center at (0.25, 0.75)
- Quadrant 8 (bottom-middle): Center at (0.50, 0.75)
- Quadrant 9 (bottom-right): Center at (0.75, 0.75)
Default bounding box size: 20% x 20% of frame
"""

        reference_path = self.base_dir / "QUADRANT_REFERENCE.md"
        with open(reference_path, 'w') as f:
            f.write(reference_content)

        print(f"✅ Created quadrant reference: {reference_path}")
        return reference_path

def create_p1000471_manual_annotations():
    """
    Manual annotations for P1000471.MP4
    Week 6 - Day 1 - Site 1 - Morning (21 min video)
    """
    raw_annotations = """
00:26 | YES | 7
01:44 | YES | 4
01:55 | YES | 4
02:03 | YES | 4
02:20 | YES | 4
02:26 | YES | 7
02:29 | YES | 7
02:44 | YES | 7
03:01 | YES | 7
03:09 | YES | 7
03:34 | 7
03:39 | 9
03:50 | 5
03:56 | 5
04:01 | 5
04:12 | 8
04:24 | 9
04:31 | 3
04:42 | 5
04:43 | 8
05:02 | 4
05:05 | 4
05:10 | 4
05:17 | 4
05:35 | 4
05:46 | 5
05:47 | 8, 4
05:53 | 4
05:59 | 4
06:04 | 7
06:06 | 7
06:29 | 9
06:30 | 9
06:34 | 5
06:35 | 5
06:36 | 5, 7
06:39 | 4, 7, 9
06:41 | 7
06:43 | 1
06:51 | 1
06:52 | 1, 2
06:60 | 8
07:01 | 8
07:04 | 6
07:08 | 6
07:12 | 9
07:17 | 9
07:29 | 9
07:33 | 2
07:36 | 2
08:06 | 2
08:51 | 4
09:43 | 1
09:48 | 1
10:06 | 1
10:34 | 1
10:55 | 5
11:21 | 5
11:22 | 5, 5
11:25 | 5, 5
11:30 | 5
11:52 | 6
11:55 | 5
14:01 | 7
14:05 | 7
14:12 | 7
14:34 | 8
14:38 | 8
14:39 | 8, 9
14:57 | 4
15:10 | 4
15:15 | 4
15:26 | 7
15:33 | 5
15:40 | 4
15:48 | 4, 5
16:04 | 1
16:06 | 1
16:12 | 1, 2
16:24 | 3
16:28 | 6
16:29 | 6
16:32 | 2
16:41 | 6
16:45 | 3
16:47 | 7, 5, 3
16:50 | 7, 5, 3
16:53 | 7, 5
16:58 | 7, 5
17:03 | 7, 5
17:06 | 7, 5, 6
17:09 | 7, 5, 6
17:13 | 7, 5, 6
17:18 | 7, 6
17:27 | 7, 6
17:31 | 5
17:33 | 8
17:37 | 8
17:42 | 8
17:49 | 8, 7
17:52 | 7, 7
18:07 | 7
18:12 | 7
18:18 | 4
18:20 | 4
18:23 | 4
18:28 | 4
18:31 | 4, 7
18:41 | 7, 8
18:42 | 7, 5
18:47 | 1, 5, 2
18:54 | 6
18:56 | 5, 6
19:02 | 9
19:05 | 9
19:12 | 3
19:17 | 3
19:21 | 6
19:24 | 6
19:35 | 5
19:40 | 2
19:54 | 3
20:09 | 3
20:26 | 2
20:32 | 7
21:02 | 7
"""

    manual_annotations = []
    dataset_creator = VenturaDataset(video_id="P1000471")

    for line in raw_annotations.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) < 2:
            continue
        timestamp = parts[0].strip()
        if len(parts) == 3 and parts[1].strip() == "YES":
            quadrants_str = parts[2].strip()
        elif len(parts) == 2:
            quadrants_str = parts[1].strip()
        else:
            continue
        timestamp_seconds = dataset_creator.parse_timestamp_to_seconds(timestamp)
        if timestamp_seconds is None:
            continue
        quadrants = dataset_creator.parse_quadrants(quadrants_str)
        manual_annotations.append({
            'timestamp': timestamp,
            'timestamp_seconds': timestamp_seconds,
            'quadrants': quadrants,
            'has_bee': True
        })

    total_frames = len(manual_annotations)
    total_bee_instances = sum([len(a['quadrants']) for a in manual_annotations])
    quadrant_usage = {}
    for annotation in manual_annotations:
        for quad in annotation['quadrants']:
            quadrant_usage[quad] = quadrant_usage.get(quad, 0) + 1

    print(f"📊 P1000471 MANUAL ANNOTATION SUMMARY:")
    print(f" Video: Week 6, Day 1, Site 1, Morning (21 min)")
    print(f" Total frames with bees: {total_frames}")
    print(f" Total bee instances: {total_bee_instances}")
    print(f" Average bees per frame: {total_bee_instances/total_frames:.2f}")
    print(f" Most active quadrants: {sorted(quadrant_usage.items(), key=lambda x: x[1], reverse=True)[:3]}")

    return manual_annotations

def create_p1000474_manual_annotations():
    """
    Manual annotations for P1000474.MP4
    Week 6 - Day 2 - Site 2 - Afternoon
    """
    raw_annotations = """
01:19 - 4
01:32 - 3
01:39 - 1
01:44 - 2
01:47 - 1
01:48 - 1
01:57 - 3
01:58 - 3, 4
01:59 - 2, 3
02:02 - 6
02:11 - 6, 3
02:17 - 6, 3
02:28 - 2
02:32 - 3, 6
02:36 - 3
02:41 - 3, 3
02:49 - 6
03:00 - 3
03:03 - 2
03:10 - 2
03:23 - 2
03:29 - 2
03:34 - 2
03:42 - 2
03:52 - 5
04:00 - 5
04:07 - 6
04:09 - 3
04:12 - 3
04:43 - 2
04:48 - 8
04:55 - 2
04:58 - 2
05:41 - 6
05:50 - 9
05:55 - 6
07:07 - 8
07:15 - 5
07:23 - 1
08:26 - 9
08:40 - 9
08:49 - 9
09:05 - 9
09:11 - 6
09:13 - 6
11:33 - 6
12:02 - 5
12:03 - 5
14:26 - 1
14:42 - 2
14:55 - 2
15:05 - 2
15:15 - 4
15:22 - 2, 5
15:41 - 7
15:45 - 7
15:52 - 7
15:56 - 7
16:03 - 4
16:08 - 4
16:11 - 4
16:15 - 4
16:20 - 4
16:26 - 1
16:28 - 1
16:30 - 1, 2
16:36 - 1
16:40 - 1, 2
16:43 - 2
16:46 - 2, 8
16:50 - 8
16:54 - 8
16:59 - 5
17:12 - 1
17:19 - 1
17:28 - 1
17:34 - 2
17:38 - 2
18:11 - 6
"""

    manual_annotations = []
    dataset_creator = VenturaDataset(video_id="P1000474")

    for line in raw_annotations.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('-')
        if len(parts) != 2:
            continue
        timestamp = parts[0].strip()
        quadrants_str = parts[1].strip()
        timestamp_seconds = dataset_creator.parse_timestamp_to_seconds(timestamp)
        if timestamp_seconds is None:
            continue
        quadrants = dataset_creator.parse_quadrants(quadrants_str)
        manual_annotations.append({
            'timestamp': timestamp,
            'timestamp_seconds': timestamp_seconds,
            'quadrants': quadrants,
            'has_bee': True
        })

    total_frames = len(manual_annotations)
    total_bee_instances = sum([len(a['quadrants']) for a in manual_annotations])
    quadrant_usage = {}
    for annotation in manual_annotations:
        for quad in annotation['quadrants']:
            quadrant_usage[quad] = quadrant_usage.get(quad, 0) + 1

    print(f"📊 P1000474 MANUAL ANNOTATION SUMMARY:")
    print(f" Video: Week 6, Day 2, Site 2, Afternoon")
    print(f" Total frames with bees: {total_frames}")
    print(f" Total bee instances: {total_bee_instances}")
    print(f" Average bees per frame: {total_bee_instances/total_frames:.2f}")
    print(f" Most active quadrants: {sorted(quadrant_usage.items(), key=lambda x: x[1], reverse=True)[:3]}")

    return manual_annotations

def main(video_id="P1000471"):
    """Main function - create dataset from manual annotations"""
    print(f"🌸 VENTURA MILK VETCH {video_id} OBJECT DETECTION DATASET")
    print(f"{'='*70}")

    print(f"🎬 Creating dataset for {video_id}...")
    print(f"📍 Using quadrant-based bee location system (1-9 grid)")

    # Get manual annotations
    if video_id == "P1000471":
        manual_annotations = create_p1000471_manual_annotations()
    elif video_id == "P1000474":
        manual_annotations = create_p1000474_manual_annotations()
        video_path = f"{_VIDEO_ROOT}/ventura_milkvetch/week 6/day 2/site 2/afternoon/P1000474.MP4"
    else:
        print(f"❌ Unknown video ID: {video_id}")
        return False

    # Create dataset
    dataset_creator = VenturaDataset(video_id=video_id)
    extracted_frames, config_path = dataset_creator.prepare_ventura_dataset(manual_annotations, video_path=video_path if video_id == "P1000474" else None)

    if extracted_frames:
        # Save annotation summary and quadrant reference
        summary_path = dataset_creator.save_annotation_summary(extracted_frames)
        reference_path = dataset_creator.create_quadrant_reference()

        print(f"\n✅ DATASET CREATION COMPLETE for {video_id}!")
        print(f"{'='*70}")
        print(f"Dataset location: {dataset_creator.base_dir}")
        print(f"Frames extracted: {len(extracted_frames)}")
        print(f"Total bee instances: {sum([f['bee_count'] for f in extracted_frames])}")
        print(f"Video: {video_id}.MP4")
        print(f"Plant type: Ventura Milk Vetch")

        print(f"\n📁 FOLDER STRUCTURE CREATED:")
        print(f" 📁 data/")
        print(f" ├── 📁 raw_videos/ # Place original MP4s here")
        print(f" ├── 📁 extracted_frames_{video_id}/ # {len(extracted_frames)} JPG frames")
        print(f" ├── 📁 annotations_{video_id}/ # {len(extracted_frames)} YOLO label files")
        print(f" ├── 📁 models/ # For trained model weights")
        print(f" ├── 📁 results/ # For test results")
        print(f" └── 📁 config/ # YOLO configuration")

        print(f"\n📋 FILES CREATED:")
        print(f" - Annotation summary: {summary_path}")
        print(f" - Quadrant reference: {reference_path}")
        print(f" - YOLO config: {config_path}")

        print(f"\n📍 QUADRANT SYSTEM:")
        print(f" 1=top-left 2=top-middle 3=top-right")
        print(f" 4=middle-left 5=center 6=middle-right")
        print(f" 7=bottom-left 8=bottom-middle 9=bottom-right")

        print(f"\n📋 NEXT STEPS:")
        print(f"1. Run simple_annotate.py to refine bounding boxes")
        print(f"2. Create scripts/3_train_model.py for YOLOv8 training")
        print(f"3. Create scripts/4_test_videos.py for testing multiple videos")
        print(f"4. Ensure {video_id}.MP4 is accessible")

        return True
    else:
        print(f"❌ Dataset creation failed for {video_id}")
        return False

if __name__ == "__main__":
    main(video_id="P1000474")  # Run for P1000474