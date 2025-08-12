#!/usr/bin/env python3
"""
Visualize all annotated frames with bounding boxes
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import pandas as pd

class AnnotationVisualizer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "extracted_frames"
        self.label_dir = self.data_dir / "annotations"
        self.output_dir = self.data_dir / "visualized_annotations"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Get all images and labels
        self.image_files = sorted([f for f in self.image_dir.glob("*.jpg")])
        
        # Load annotation summary
        self.load_annotation_summary()
        
        print(f"🎨 ANNOTATION VISUALIZER")
        print(f"Found {len(self.image_files)} frames to visualize")
        print(f"Output directory: {self.output_dir}")
    
    def load_annotation_summary(self):
        """Load annotation summary for context"""
        summary_path = self.data_dir / "annotation_summary.csv"
        self.summary = {}
        
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            for _, row in df.iterrows():
                filename = row['frame_filename']
                self.summary[filename] = {
                    'timestamp': row['timestamp'],
                    'bee_count': row['bee_count'],
                    'quadrants': row['quadrants']
                }
    
    def get_frame_info(self, filename):
        """Get frame info"""
        if filename in self.summary:
            info = self.summary[filename]
            return f"Time: {info['timestamp']} | Expected: {info['bee_count']} | Quadrants: {info['quadrants']}"
        return "No info"
    
    def load_annotations(self, image_path):
        """Load YOLO annotations for an image"""
        label_path = self.label_dir / (image_path.stem + ".txt")
        boxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            parts = line.split()
                            if len(parts) == 5:
                                _, cx, cy, w, h = map(float, parts)
                                boxes.append([cx, cy, w, h])
                        except ValueError:
                            pass
        
        return boxes
    
    def draw_single_frame(self, image_path, save_individual=True):
        """Draw bounding boxes on a single frame"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        
        # Load annotations
        boxes = self.load_annotations(image_path)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        
        # Draw bounding boxes
        for i, (cx, cy, w, h) in enumerate(boxes):
            # Convert from YOLO format to pixel coordinates
            x = (cx - w/2) * img_w
            y = (cy - h/2) * img_h
            width = w * img_w
            height = h * img_h
            
            # Draw rectangle
            rect = patches.Rectangle((x, y), width, height, 
                                   linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add bee label
            ax.text(x, y-10, f'Bee {i+1}', color='red', fontweight='bold', 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        # Add title with info
        frame_info = self.get_frame_info(image_path.name)
        title = f"{image_path.name}\n{frame_info}\nDetected: {len(boxes)} bees"
        ax.set_title(title, fontsize=10, pad=15)
        ax.axis('off')
        
        if save_individual:
            # Save individual annotated image
            output_path = self.output_dir / f"annotated_{image_path.stem}.jpg"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            return fig, ax
    
    def create_overview_grid(self, images_per_row=4, max_images=None, frames_per_grid=30):
        """Create grid overview(s) of all annotated frames"""
        print("🖼️ Creating overview grids...")
        
        images_to_show = self.image_files[:max_images] if max_images else self.image_files
        total_images = len(images_to_show)
        
        # Create multiple grids if we have many images
        num_grids = (total_images + frames_per_grid - 1) // frames_per_grid
        grid_files = []
        
        for grid_idx in range(num_grids):
            start_idx = grid_idx * frames_per_grid
            end_idx = min(start_idx + frames_per_grid, total_images)
            current_batch = images_to_show[start_idx:end_idx]
            
            rows = (len(current_batch) + images_per_row - 1) // images_per_row
            
            fig, axes = plt.subplots(rows, images_per_row, figsize=(20, 5*rows))
            
            if num_grids > 1:
                fig.suptitle(f'Ventura Milkvetch Bee Annotations - Grid {grid_idx + 1}/{num_grids}\nFrames {start_idx + 1}-{end_idx} of {total_images}', fontsize=16)
            else:
                fig.suptitle(f'Ventura Milkvetch Bee Annotations Overview\n{total_images} Frames Total', fontsize=16)
            
            # Handle single row case
            if rows == 1:
                if images_per_row == 1:
                    axes = [axes]
                else:
                    axes = axes.reshape(1, -1)
            elif images_per_row == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, image_path in enumerate(current_batch):
                row = idx // images_per_row
                col = idx % images_per_row
                
                # Load and process image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_h, img_w = image.shape[:2]
                
                # Load annotations
                boxes = self.load_annotations(image_path)
                
                # Plot
                ax = axes[row, col] if rows > 1 or images_per_row > 1 else axes[idx]
                ax.imshow(image)
                
                # Draw boxes
                for i, (cx, cy, w, h) in enumerate(boxes):
                    x = (cx - w/2) * img_w
                    y = (cy - h/2) * img_h
                    width = w * img_w
                    height = h * img_h
                    
                    rect = patches.Rectangle((x, y), width, height, 
                                           linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                
                # Title
                frame_info = self.get_frame_info(image_path.name)
                ax.set_title(f"Frame {start_idx + idx + 1}\n{len(boxes)} bees", fontsize=8)
                ax.axis('off')
            
            # Hide empty subplots
            for idx in range(len(current_batch), rows * images_per_row):
                row = idx // images_per_row
                col = idx % images_per_row
                if rows > 1 or images_per_row > 1:
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # Save grid
            if num_grids > 1:
                grid_path = self.output_dir / f"overview_grid_{grid_idx + 1}_of_{num_grids}.jpg"
            else:
                grid_path = self.output_dir / "overview_grid.jpg"
                
            plt.savefig(grid_path, dpi=200, bbox_inches='tight')
            plt.show()
            grid_files.append(grid_path)
            
            print(f"✅ Saved grid {grid_idx + 1}/{num_grids}: {grid_path}")
        
        return grid_files
    
    def create_individual_images(self):
        """Create individual annotated images"""
        print("🎨 Creating individual annotated images...")
        
        created_files = []
        for i, image_path in enumerate(self.image_files):
            print(f"Processing {i+1}/{len(self.image_files)}: {image_path.name}")
            
            output_path = self.draw_single_frame(image_path, save_individual=True)
            if output_path:
                created_files.append(output_path)
        
        print(f"✅ Created {len(created_files)} individual annotated images")
        return created_files
    
    def create_stats_summary(self):
        """Create statistics summary of annotations"""
        print("📊 Creating annotation statistics...")
        
        total_frames = len(self.image_files)
        total_bees = 0
        frames_with_bees = 0
        bee_counts = []
        
        for image_path in self.image_files:
            boxes = self.load_annotations(image_path)
            bee_count = len(boxes)
            total_bees += bee_count
            
            if bee_count > 0:
                frames_with_bees += 1
            
            bee_counts.append(bee_count)
        
        # Create statistics plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Annotation Statistics Summary', fontsize=16)
        
        # Bee count distribution
        ax1.hist(bee_counts, bins=range(max(bee_counts)+2), alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Number of Bees per Frame')
        ax1.set_ylabel('Number of Frames')
        ax1.set_title('Distribution of Bee Counts')
        ax1.grid(True, alpha=0.3)
        
        # Bees over time
        ax2.plot(range(1, len(bee_counts)+1), bee_counts, marker='o', markersize=3)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Number of Bees')
        ax2.set_title('Bee Counts Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f"""
        Total Frames: {total_frames}
        Frames with Bees: {frames_with_bees}
        Total Bee Instances: {total_bees}
        Average Bees per Frame: {total_bees/total_frames:.2f}
        Max Bees in Single Frame: {max(bee_counts)}
        Frames with 0 bees: {bee_counts.count(0)}
        Frames with 1 bee: {bee_counts.count(1)}
        Frames with 2+ bees: {sum(1 for x in bee_counts if x >= 2)}
        """
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
        ax3.set_title('Summary Statistics')
        ax3.axis('off')
        
        # Timeline with timestamps
        if self.summary:
            timestamps = []
            counts = []
            for i, image_path in enumerate(self.image_files):
                if image_path.name in self.summary:
                    timestamp = self.summary[image_path.name]['timestamp']
                    timestamps.append(timestamp)
                    counts.append(bee_counts[i])
            
            if timestamps:
                ax4.scatter(timestamps, counts, alpha=0.6)
                ax4.set_xlabel('Timestamp (MM:SS)')
                ax4.set_ylabel('Number of Bees')
                ax4.set_title('Bee Activity Over Video Timeline')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save statistics
        stats_path = self.output_dir / "annotation_statistics.jpg"
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved statistics summary: {stats_path}")
        return stats_path
    
    def run_full_visualization(self):
        """Run complete visualization pipeline"""
        print(f"\n🎨 STARTING FULL ANNOTATION VISUALIZATION")
        print(f"{'='*60}")
        
        # Create individual images
        individual_files = self.create_individual_images()
        
        # Create overview grids (all frames, broken into manageable grids)
        overview_files = self.create_overview_grid(images_per_row=6, frames_per_grid=30)
        
        # Create statistics
        stats_file = self.create_stats_summary()
        
        print(f"\n✅ VISUALIZATION COMPLETE!")
        print(f"{'='*60}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Individual images: {len(individual_files)} files")
        print(f"🖼️ Overview grids: {len(overview_files)} files ({overview_files[0].name} etc.)")
        print(f"📈 Statistics: {stats_file}")
        print(f"\n💡 You can now:")
        print(f"   • Browse individual annotated images in {self.output_dir}")
        print(f"   • View the overview grid to see all frames at once")
        print(f"   • Check statistics to validate your annotation quality")


def main():
    """Run annotation visualization"""
    data_dir = "data"
    
    if not os.path.exists(f"{data_dir}/extracted_frames"):
        print("❌ Frames directory not found")
        return
    
    if not os.path.exists(f"{data_dir}/annotations"):
        print("❌ Annotations directory not found")
        return
    
    visualizer = AnnotationVisualizer(data_dir)
    visualizer.run_full_visualization()

if __name__ == "__main__":
    main()