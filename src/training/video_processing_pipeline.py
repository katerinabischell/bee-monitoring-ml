"""
Video Processing Pipeline for Bombus Detection
Author: Katerina Bischel
Project: Endangered Coastal Plant Pollinator Monitoring

This script processes your video files and creates a labeled dataset for training.
"""

import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import shutil
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Processes videos and extracts labeled frames for ML training"""
    
    def __init__(self, 
                 video_root_dir="data/raw",
                 observations_file="Collection Observations3.xlsx", 
                 output_dir="data/processed"):
        
        self.video_root = Path(video_root_dir)
        self.observations_file = observations_file
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.frames_dir = self.output_dir / "frames"
        self.positive_dir = self.frames_dir / "positive"
        self.negative_dir = self.frames_dir / "negative"
        self.annotations_dir = self.output_dir / "annotations"
        
        for dir_path in [self.positive_dir, self.negative_dir, self.annotations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load and process observation data
        self.observations_df = None
        self.load_observations()
        
        # Processing statistics
        self.stats = {
            'videos_processed': 0,
            'videos_failed': 0,
            'frames_extracted': 0,
            'positive_frames': 0,
            'negative_frames': 0,
            'processing_errors': []
        }
    
    def load_observations(self):
        """Load and clean observation data"""
        try:
            # Try both sheets, combine them
            birds_beak_df = pd.read_excel(self.observations_file, sheet_name='Birds Beak')
            milkvetch_df = pd.read_excel(self.observations_file, sheet_name='Ventura Milkvetch')
            
            # Add plant type column
            birds_beak_df['Plant_Type'] = 'Birds_Beak'
            milkvetch_df['Plant_Type'] = 'Ventura_Milkvetch'
            
            # Combine dataframes
            self.observations_df = pd.concat([birds_beak_df, milkvetch_df], ignore_index=True)
            
            # Clean activity columns
            self.observations_df['Activity on site'] = self.observations_df['Activity on site'].fillna('none').astype(str).str.lower()
            self.observations_df['Activity around '] = self.observations_df['Activity around '].fillna('none').astype(str).str.lower()
            self.observations_df['Notes'] = self.observations_df['Notes'].fillna('').astype(str)
            
            logger.info(f"Loaded {len(self.observations_df)} observation records")
            
        except Exception as e:
            logger.error(f"Failed to load observations: {e}")
            raise
    
    def determine_video_label(self, week, day, site, shift, plant_type, camera=1):
        """
        Determine if a video contains bombus activity based on observations
        Returns: ('positive'/'negative', confidence_score, species_info)
        """
        
        # Filter observations for this specific video
        # Convert week to string for comparison (handle both "2" and "2b" formats)
        week_str = str(week)
        
        mask = (
            (self.observations_df['Week '].astype(str).str.contains(week_str, na=False)) &
            (self.observations_df['Site #'] == site) &
            (self.observations_df['Camera #'] == camera) &
            (self.observations_df['Shift type'].str.lower() == shift.lower()) &
            (self.observations_df['Plant_Type'].str.replace(' ', '_') == plant_type)
        )
        
        matching_obs = self.observations_df[mask]
        
        if len(matching_obs) == 0:
            # Try without camera filter (in case camera info is missing/different)
            mask_no_camera = (
                (self.observations_df['Week '].astype(str).str.contains(week_str, na=False)) &
                (self.observations_df['Site #'] == site) &
                (self.observations_df['Shift type'].str.lower() == shift.lower()) &
                (self.observations_df['Plant_Type'].str.replace(' ', '_') == plant_type)
            )
            matching_obs = self.observations_df[mask_no_camera]
        
        if len(matching_obs) == 0:
            logger.warning(f"No observations found for week:{week}, day:{day}, site:{site}, camera:{camera}, shift:{shift}, plant:{plant_type}")
            return 'negative', 0.0, {}
        
        # Check for bombus activity
        bombus_keywords = ['bombus', 'b. californicus', 'b. vosnesenskii', 'bombus v', 'bombus vos']
        
        for _, obs in matching_obs.iterrows():
            activity_on = str(obs['Activity on site']).lower()
            activity_around = str(obs['Activity around ']).lower()
            
            # Check for bombus mentions
            bombus_on_site = any(keyword in activity_on for keyword in bombus_keywords)
            bombus_around = any(keyword in activity_around for keyword in bombus_keywords)
            
            if bombus_on_site or bombus_around:
                # Determine species
                species_info = {}
                if any(keyword in activity_on or keyword in activity_around 
                      for keyword in ['vosnesenskii', 'bombus v', 'bombus vos']):
                    species_info['vosnesenskii'] = True
                if 'californicus' in activity_on or 'californicus' in activity_around:
                    species_info['californicus'] = True
                
                # Higher confidence if bombus is on the plant vs just around
                confidence = 0.9 if bombus_on_site else 0.7
                
                logger.info(f"✅ POSITIVE: Week {week}, Site {site}, {shift} - {species_info}")
                return 'positive', confidence, species_info
        
        return 'negative', 0.8, {}
    
    def extract_frames_from_video(self, video_path, num_frames=30, method='uniform'):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            method: 'uniform' for evenly spaced, 'random' for random sampling
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            if total_frames == 0:
                logger.warning(f"Video has no frames: {video_path}")
                return []
            
            # Adjust number of frames based on video length
            # For shorter videos, extract fewer frames
            if duration < 300:  # Less than 5 minutes
                num_frames = min(num_frames, max(10, int(duration / 10)))
            
            frames = []
            
            if method == 'uniform':
                if total_frames <= num_frames:
                    frame_indices = list(range(0, total_frames, max(1, total_frames // num_frames)))
                else:
                    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            else:  # random
                frame_indices = np.random.choice(total_frames, min(num_frames, total_frames), replace=False)
                frame_indices.sort()
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Resize frame to standard size
                    frame_resized = cv2.resize(frame, (224, 224))
                    frames.append({
                        'frame': frame_resized,
                        'frame_number': frame_idx,
                        'timestamp': frame_idx / fps if fps > 0 else 0
                    })
                
                if len(frames) >= num_frames:
                    break
            
            logger.info(f"Extracted {len(frames)} frames from {video_path.name} (duration: {duration:.1f}s)")
            return frames
            
        finally:
            cap.release()
    
    def process_single_video(self, video_path, metadata):
        """Process a single video file"""
        
        try:
            # Determine label based on observations
            label, confidence, species_info = self.determine_video_label(
                metadata['week'], metadata.get('day', 1), metadata['site'], 
                metadata['shift'], metadata['plant_type'], metadata.get('camera', 1)
            )
            
            # Extract frames
            frames = self.extract_frames_from_video(video_path, num_frames=25)
            
            if not frames:
                logger.warning(f"No frames extracted from {video_path}")
                return False
            
            # Save frames to appropriate directory
            output_subdir = self.positive_dir if label == 'positive' else self.negative_dir
            
            saved_frames = []
            for i, frame_data in enumerate(frames):
                # Create filename with metadata
                filename = f"{metadata['plant_type']}_w{metadata['week']}_d{metadata.get('day', 1)}_s{metadata['site']}_c{metadata.get('camera', 1)}_{metadata['shift']}_f{i:03d}.jpg"
                frame_path = output_subdir / filename
                
                # Save frame
                success = cv2.imwrite(str(frame_path), frame_data['frame'])
                if success:
                    saved_frames.append({
                        'filename': filename,
                        'label': label,
                        'confidence': confidence,
                        'species_info': species_info,
                        'frame_number': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp'],
                        'video_source': str(video_path),
                        'metadata': metadata
                    })
                    
                    # Update statistics
                    if label == 'positive':
                        self.stats['positive_frames'] += 1
                    else:
                        self.stats['negative_frames'] += 1
                    
                    self.stats['frames_extracted'] += 1
            
            logger.info(f"Processed {video_path.name}: {label} ({len(saved_frames)} frames)")
            self.stats['videos_processed'] += 1
            
            return saved_frames
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            self.stats['videos_failed'] += 1
            self.stats['processing_errors'].append(str(e))
            return False
    
    def find_video_files(self):
        """Find all video files in the directory structure"""
        
        video_files = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        logger.info(f"Searching for videos in: {self.video_root}")
        
        if not self.video_root.exists():
            logger.error(f"Video directory not found: {self.video_root}")
            return []
        
        # Walk through directory structure
        for video_file in self.video_root.rglob('*'):
            if video_file.suffix.lower() in video_extensions:
                
                # Parse metadata from path
                try:
                    path_parts = video_file.parts
                    
                    # Try to extract metadata from file path
                    # Look for patterns like: week_X/day_Y/site_Z/shift_type.mp4
                    metadata = self.parse_video_metadata(video_file)
                    
                    if metadata:
                        video_files.append((video_file, metadata))
                        logger.debug(f"Found video: {video_file} -> {metadata}")
                    else:
                        logger.warning(f"Could not parse metadata for: {video_file}")
                
                except Exception as e:
                    logger.warning(f"Error parsing {video_file}: {e}")
        
        logger.info(f"Found {len(video_files)} video files")
        return video_files
    
    def parse_video_metadata(self, video_path):
        """
        Parse metadata from video file path
        Expected structure: birds_beak/week_X/day_Y/site_Z/shift_type/filename.MP4
        """
        
        path_parts = video_path.parts
        filename = video_path.stem
        
        metadata = {}
        
        # Parse from path structure
        try:
            # Find relevant parts in the path
            for i, part in enumerate(path_parts):
                part_lower = part.lower()
                
                # Plant type detection
                if part_lower == 'birds_beak':
                    metadata['plant_type'] = 'Birds_Beak'
                elif part_lower == 'ventura_milkvetch' or 'milkvetch' in part_lower:
                    metadata['plant_type'] = 'Ventura_Milkvetch'
                
                # Week detection (week_2, week_3, etc.)
                elif part_lower.startswith('week_'):
                    week_num = part_lower.replace('week_', '')
                    metadata['week'] = week_num
                
                # Day detection (day_1, day_2, etc.)
                elif part_lower.startswith('day_'):
                    day_num = int(part_lower.replace('day_', ''))
                    metadata['day'] = day_num
                
                # Site detection (site_1, site_2, etc.)
                elif part_lower.startswith('site_'):
                    site_num = int(part_lower.replace('site_', ''))
                    metadata['site'] = site_num
                
                # Shift detection (morning, mid, afternoon)
                elif part_lower in ['morning', 'mid', 'afternoon']:
                    metadata['shift'] = part_lower.capitalize()
            
            # Default camera number (since not in path structure)
            # We'll assume camera 1 unless we can detect from filename
            metadata['camera'] = 1
            
            # Try to detect camera from filename if there are multiple files
            # P1000372.MP4, P1000373.MP4 might indicate different cameras or time segments
            filename_lower = filename.lower()
            if 'p1000373' in filename_lower or any(x in filename_lower for x in ['cam2', 'camera2', 'c2']):
                metadata['camera'] = 2
            
        except Exception as e:
            logger.warning(f"Error parsing path {video_path}: {e}")
            return None
        
        # Validate we have minimum required metadata
        required_fields = ['week', 'day', 'site', 'shift', 'plant_type']
        if all(field in metadata for field in required_fields):
            return metadata
        else:
            missing = [field for field in required_fields if field not in metadata]
            logger.warning(f"Missing metadata fields for {video_path}: {missing}")
            logger.warning(f"Path parts: {path_parts}")
            return None
    
    def process_video_subset(self, max_videos=50):
        """Process a subset of videos for initial testing"""
        
        logger.info(f"Processing subset of up to {max_videos} videos...")
        
        # Find all video files
        video_files = self.find_video_files()
        
        if not video_files:
            logger.error("No video files found!")
            return False
        
        # Stratify selection to get both positive and negative samples
        positive_videos = []
        negative_videos = []
        
        for video_path, metadata in video_files:
            # Filter for weeks 2-4 only
            try:
                week_num = int(str(metadata['week']).replace('a', '').replace('b', ''))
                if week_num not in [2, 3, 4]:
                    logger.info(f"Skipping week {metadata['week']} (only processing weeks 2-4)")
                    continue
            except:
                logger.warning(f"Could not parse week number from {metadata['week']}")
                continue
            
            label, confidence, species_info = self.determine_video_label(
                metadata['week'], metadata.get('day', 1), metadata['site'], 
                metadata['shift'], metadata['plant_type'], metadata.get('camera', 1)
            )
            
            if label == 'positive':
                positive_videos.append((video_path, metadata))
            else:
                negative_videos.append((video_path, metadata))
        
        # Select balanced subset
        max_positive = min(len(positive_videos), max_videos // 3)  # 1/3 positive
        max_negative = min(len(negative_videos), max_videos - max_positive)
        
        selected_videos = positive_videos[:max_positive] + negative_videos[:max_negative]
        
        logger.info(f"Selected {len(selected_videos)} videos ({max_positive} positive, {max_negative} negative)")
        
        # Process selected videos
        all_frame_data = []
        for i, (video_path, metadata) in enumerate(selected_videos):
            logger.info(f"Processing video {i+1}/{len(selected_videos)}: {video_path.name}")
            
            frame_data = self.process_single_video(video_path, metadata)
            if frame_data:
                all_frame_data.extend(frame_data)
        
        # Save results
        if all_frame_data:
            frame_df = pd.DataFrame(all_frame_data)
            frame_df.to_csv(self.annotations_dir / "frame_annotations_subset.csv", index=False)
            self.save_processing_summary(all_frame_data)
        
        return True
    
    def process_positive_videos_only(self):
        """Process only videos with bombus activity"""
        
        logger.info("Processing only videos with bombus activity...")
        
        # Find all video files
        video_files = self.find_video_files()
        
        if not video_files:
            logger.error("No video files found!")
            return False
        
        # Filter for positive videos only
        positive_videos = []
        
        for video_path, metadata in video_files:
            label, confidence, species_info = self.determine_video_label(
                metadata['week'], metadata['site'], metadata['camera'], 
                metadata['shift'], metadata['plant_type']
            )
            
            if label == 'positive':
                positive_videos.append((video_path, metadata))
        
        logger.info(f"Found {len(positive_videos)} videos with bombus activity")
        
        # Process positive videos
        all_frame_data = []
        for i, (video_path, metadata) in enumerate(positive_videos):
            logger.info(f"Processing positive video {i+1}/{len(positive_videos)}: {video_path.name}")
            
            frame_data = self.process_single_video(video_path, metadata)
            if frame_data:
                all_frame_data.extend(frame_data)
        
        # Save results
        if all_frame_data:
            frame_df = pd.DataFrame(all_frame_data)
            frame_df.to_csv(self.annotations_dir / "frame_annotations_positive_only.csv", index=False)
            self.save_processing_summary(all_frame_data)
        
        return True
        """Process all videos in the directory structure"""
        
        logger.info("Starting video processing pipeline...")
        
        # Find all video files
        video_files = self.find_video_files()
        
        if not video_files:
            logger.error("No video files found! Check your directory structure.")
            return False
        
        # Process each video
        all_frame_data = []
        
        for i, (video_path, metadata) in enumerate(video_files):
            logger.info(f"Processing video {i+1}/{len(video_files)}: {video_path.name}")
            
            frame_data = self.process_single_video(video_path, metadata)
            if frame_data:
                all_frame_data.extend(frame_data)
        
        # Save frame metadata
        if all_frame_data:
            frame_df = pd.DataFrame(all_frame_data)
            frame_df.to_csv(self.annotations_dir / "frame_annotations.csv", index=False)
            
            # Save processing summary
            self.save_processing_summary(all_frame_data)
        
        logger.info("Video processing complete!")
        return True
    
    def save_processing_summary(self, frame_data):
        """Save processing summary and statistics"""
        
        summary = {
            'processing_date': datetime.now().isoformat(),
            'statistics': self.stats,
            'total_frames': len(frame_data),
            'class_distribution': {
                'positive': self.stats['positive_frames'],
                'negative': self.stats['negative_frames']
            },
            'positive_ratio': self.stats['positive_frames'] / max(1, self.stats['frames_extracted'])
        }
        
        # Save summary JSON
        with open(self.annotations_dir / "processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n" + "="*60)
        print(f"🎬 VIDEO PROCESSING COMPLETE")
        print(f"="*60)
        print(f"📊 Videos processed: {self.stats['videos_processed']}")
        print(f"❌ Videos failed: {self.stats['videos_failed']}")
        print(f"🖼️  Total frames extracted: {self.stats['frames_extracted']}")
        print(f"✅ Positive frames (with bombus): {self.stats['positive_frames']}")
        print(f"❌ Negative frames (no bombus): {self.stats['negative_frames']}")
        print(f"📈 Positive ratio: {summary['positive_ratio']:.1%}")
        print(f"📁 Frames saved to: {self.frames_dir}")
        print(f"📋 Annotations saved to: {self.annotations_dir}")

def main():
    """Main processing pipeline"""
    
    print("🎬 Video Processing Pipeline for Bombus Detection")
    print("="*60)
    
    # Ask user for video directory location
    print("📁 Video Directory Options:")
    print("   1. External drive path (e.g., /Volumes/Expansion/summer2025_ncos_kb_collections)")
    print("   2. Local directory (data/raw)")
    print("   3. Custom path")
    
    video_path = input("\nEnter the path to your videos (or press Enter for option 1): ").strip()
    
    if not video_path:
        # Default to likely external drive path
        video_path = "/Volumes/Expansion/summer2025_ncos_kb_collections"
        print(f"Using default path: {video_path}")
    
    video_dir = Path(video_path)
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        print(f"📁 Please check that your external drive is mounted and the path is correct")
        print(f"💡 Try: ls /Volumes/ to see available drives")
        return
    
    # Check if observations file exists
    obs_file = "Collection Observations3.xlsx"
    if not Path(obs_file).exists():
        print(f"❌ Observations file not found: {obs_file}")
        print(f"📄 Please ensure your Excel file is in the current directory")
        return
    
    try:
        # Initialize processor with custom video path
        processor = VideoProcessor(
            video_root_dir=str(video_dir),
            observations_file=obs_file,
            output_dir="data/processed"
        )
        
        # Ask user about processing options
        print(f"\n🎯 Processing Options:")
        print(f"   1. Process ALL videos (may take hours)")
        print(f"   2. Process SUBSET for testing (recommended)")
        print(f"   3. Process only POSITIVE samples (videos with bombus)")
        
        choice = input("\nChoose option (1-3, or Enter for subset): ").strip()
        
        if choice == "1":
            # Process all videos
            success = processor.process_all_videos()
        elif choice == "3":
            # Process only positive samples
            success = processor.process_positive_videos_only()
        else:
            # Process subset (default)
            success = processor.process_video_subset(max_videos=50)
        
        if success:
            print(f"\n✅ Processing completed successfully!")
            print(f"🚀 Next step: Train your machine learning model using the extracted frames")
        else:
            print(f"\n❌ Processing completed with errors. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()