#!/usr/bin/env python3
"""
Matplotlib-based annotation GUI for refining bee bounding boxes
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import json
from pathlib import Path
import pandas as pd

class VenturaBeeAnnotator:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "extracted_frames"
        self.label_dir = self.data_dir / "annotations"
        
        # Get all JPG files and sort them
        self.image_files = sorted([f for f in self.image_dir.glob("*.jpg")])
        
        self.current_idx = 0
        self.boxes = []
        self.fig = None
        self.ax = None
        self.current_image = None
        self.start_x = None
        self.start_y = None
        
        # Load annotation summary for context
        self.load_annotation_summary()
        
        print(f"🎯 VENTURA BEE ANNOTATOR")
        print(f"{'='*50}")
        print(f"Found {len(self.image_files)} frames to annotate")
        print(f"Images: {self.image_dir}")
        print(f"Labels: {self.label_dir}")
        
    def load_annotation_summary(self):
        """Load the annotation summary for context"""
        summary_path = self.data_dir / "annotation_summary.csv"
        self.summary = {}
        
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            for _, row in df.iterrows():
                filename = row['frame_filename']
                self.summary[filename] = {
                    'timestamp': row['timestamp'],
                    'bee_count': row['bee_count'],
                    'quadrants': row['quadrants'],
                    'quadrant_names': row.get('quadrant_names', '')
                }
            print(f"✅ Loaded annotation summary with {len(self.summary)} entries")
        else:
            print("⚠️ No annotation summary found")
        
    def get_frame_info(self, filename):
        """Get info about current frame from summary"""
        if filename in self.summary:
            info = self.summary[filename]
            return f"Time: {info['timestamp']} | Expected bees: {info['bee_count']} | Quadrants: {info['quadrants']}"
        return "No summary info available"
        
    def load_image(self):
        """Load current image and its annotations"""
        if self.current_idx >= len(self.image_files):
            print("✅ All images annotated!")
            return False
            
        image_path = self.image_files[self.current_idx]
        self.current_image = cv2.imread(str(image_path))
        
        if self.current_image is None:
            print(f"❌ Could not load image: {image_path}")
            return False
            
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Load existing YOLO annotations
        label_path = self.label_dir / (image_path.stem + ".txt")
        self.boxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            try:
                                class_id, cx, cy, w, h = map(float, parts)
                                # Convert from YOLO format (normalized) to pixel coordinates
                                img_h, img_w = self.current_image.shape[:2]
                                x = (cx - w/2) * img_w
                                y = (cy - h/2) * img_h
                                width = w * img_w
                                height = h * img_h
                                self.boxes.append([x, y, width, height])
                            except ValueError:
                                print(f"⚠️ Invalid annotation line: {line}")
            
            print(f"📦 Loaded {len(self.boxes)} existing bounding boxes for {image_path.name}")
        else:
            print(f"📝 No existing annotations for {image_path.name}")
        
        return True
        
    def display_image(self):
        """Display image with current annotations"""
        if self.fig:
            plt.close(self.fig)
            
        # Create figure with extra space for info
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        self.ax.imshow(self.current_image)
        
        # Draw existing bounding boxes
        for i, (x, y, w, h) in enumerate(self.boxes):
            rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
            self.ax.add_patch(rect)
            # Add bee number label
            self.ax.text(x, y-5, f'Bee {i+1}', color='red', fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # Title with frame info
        filename = self.image_files[self.current_idx].name
        frame_info = self.get_frame_info(filename)
        title = f"Frame {self.current_idx + 1}/{len(self.image_files)}: {filename}\n{frame_info}"
        self.ax.set_title(title, fontsize=12, pad=20)
        self.ax.axis('off')
        
        # Connect mouse events for drawing boxes
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        # Add control buttons at bottom
        button_width = 0.08
        button_height = 0.04
        button_y = 0.01
        
        ax_prev = plt.axes([0.1, button_y, button_width, button_height])
        ax_next = plt.axes([0.85, button_y, button_width, button_height])
        ax_save = plt.axes([0.76, button_y, button_width, button_height])
        ax_clear = plt.axes([0.67, button_y, button_width, button_height])
        ax_undo = plt.axes([0.58, button_y, button_width, button_height])
        ax_auto = plt.axes([0.49, button_y, button_width, button_height])
        
        self.btn_prev = Button(ax_prev, '← Prev')
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_auto = Button(ax_auto, 'Auto-fit')
        
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_save.on_clicked(self.save_annotations)
        self.btn_clear.on_clicked(self.clear_boxes)
        self.btn_undo.on_clicked(self.undo_last)
        self.btn_auto.on_clicked(self.auto_fit_boxes)
        
        # Add instructions text
        instructions = ("INSTRUCTIONS: Click & drag to draw bounding boxes around bees\n" +
                       "Use buttons: Auto-fit (resize boxes) | Undo (remove last) | Clear All | Save | Next →")
        self.fig.text(0.5, 0.08, instructions, ha='center', fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for buttons and instructions
        plt.show()
    
    def on_press(self, event):
        """Handle mouse press - start drawing box"""
        if event.inaxes != self.ax:
            return
        self.start_x = event.xdata
        self.start_y = event.ydata
        
    def on_release(self, event):
        """Handle mouse release - finish drawing box"""
        if event.inaxes != self.ax or self.start_x is None:
            return
            
        end_x = event.xdata
        end_y = event.ydata
        
        # Calculate box coordinates
        x = min(self.start_x, end_x)
        y = min(self.start_y, end_y)
        width = abs(end_x - self.start_x)
        height = abs(end_y - self.start_y)
        
        # Only add if box is reasonably sized
        if width > 5 and height > 5:
            self.boxes.append([x, y, width, height])
            print(f"✅ Added bee box #{len(self.boxes)}: ({x:.1f}, {y:.1f}) {width:.1f}x{height:.1f}")
            self.display_image()  # Refresh display
        
        # Reset drawing state
        self.start_x = None
        self.start_y = None
    
    def auto_fit_boxes(self, event=None):
        """Auto-fit existing boxes to reasonable bee sizes"""
        if not self.boxes:
            print("No boxes to auto-fit")
            return
            
        img_h, img_w = self.current_image.shape[:2]
        
        # Reasonable bee size range (as fraction of image)
        min_size = min(img_w, img_h) * 0.02  # 2% of smaller dimension
        max_size = min(img_w, img_h) * 0.25  # 25% of smaller dimension
        
        for i, (x, y, w, h) in enumerate(self.boxes):
            # Clamp box size to reasonable range
            new_w = max(min_size, min(max_size, w))
            new_h = max(min_size, min(max_size, h))
            
            # Keep box centered
            center_x = x + w/2
            center_y = y + h/2
            new_x = center_x - new_w/2
            new_y = center_y - new_h/2
            
            # Keep box within image bounds
            new_x = max(0, min(img_w - new_w, new_x))
            new_y = max(0, min(img_h - new_h, new_y))
            
            self.boxes[i] = [new_x, new_y, new_w, new_h]
        
        print(f"🔧 Auto-fitted {len(self.boxes)} bounding boxes")
        self.display_image()
    
    def save_annotations(self, event=None):
        """Save current annotations to YOLO format"""
        image_path = self.image_files[self.current_idx]
        label_path = self.label_dir / (image_path.stem + ".txt")
        
        img_h, img_w = self.current_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for x, y, w, h in self.boxes:
                # Convert pixel coordinates to YOLO format (normalized)
                center_x = (x + w/2) / img_w
                center_y = (y + h/2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # Clamp values to [0, 1] range
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))
                
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        print(f"💾 Saved {len(self.boxes)} annotations to {label_path.name}")
        
    def next_image(self, event=None):
        """Go to next image"""
        self.save_annotations()  # Auto-save before moving
        self.current_idx += 1
        if self.current_idx >= len(self.image_files):
            plt.close('all')
            print(f"\n🎉 ANNOTATION COMPLETE!")
            print(f"✅ Annotated all {len(self.image_files)} frames")
            print(f"📁 Annotations saved in: {self.label_dir}")
            return
            
        if self.load_image():
            self.display_image()
    
    def prev_image(self, event=None):
        """Go to previous image"""
        if self.current_idx > 0:
            self.save_annotations()  # Auto-save before moving
            self.current_idx -= 1
            if self.load_image():
                self.display_image()
        else:
            print("Already at first image")
    
    def clear_boxes(self, event=None):
        """Clear all bounding boxes"""
        self.boxes = []
        self.display_image()
        print("🗑️ Cleared all bounding boxes")
    
    def undo_last(self, event=None):
        """Remove the last bounding box"""
        if self.boxes:
            removed = self.boxes.pop()
            self.display_image()
            print(f"↶ Undid last box: ({removed[0]:.1f}, {removed[1]:.1f}) {removed[2]:.1f}x{removed[3]:.1f}")
        else:
            print("No boxes to undo")
    
    def start_annotation(self):
        """Start the annotation process"""
        if len(self.image_files) == 0:
            print("❌ No JPG files found in extracted_frames directory!")
            return
            
        print(f"\n🎯 STARTING VENTURA BEE ANNOTATION")
        print(f"{'='*60}")
        print(f"📊 Dataset Stats:")
        print(f"   Total frames: {len(self.image_files)}")
        print(f"   Data directory: {self.data_dir}")
        print(f"\n🖱️  CONTROLS:")
        print(f"   • Click & drag: Draw bounding box around bee")
        print(f"   • Auto-fit: Resize boxes to reasonable bee sizes")  
        print(f"   • Undo: Remove last drawn box")
        print(f"   • Clear All: Remove all boxes")
        print(f"   • Save: Save current annotations")
        print(f"   • Next/Prev: Navigate between frames (auto-saves)")
        print(f"\n💡 TIPS:")
        print(f"   • Each frame shows expected bee count and quadrants")
        print(f"   • Use Auto-fit to quickly resize boxes to good bee sizes")
        print(f"   • Annotations are saved automatically when you navigate")
        print(f"{'='*60}")
        
        # Start with first image
        if self.load_image():
            self.display_image()
        else:
            print("❌ Could not load first image")


def main():
    """Run the annotation tool"""
    data_dir = "data"
    
    # Check if required directories exist
    if not os.path.exists(f"{data_dir}/extracted_frames"):
        print(f"❌ Frames directory not found: {data_dir}/extracted_frames")
        print("💡 Make sure you've run 1_extract_frames.py first")
        return
    
    if not os.path.exists(f"{data_dir}/annotations"):
        print(f"❌ Annotations directory not found: {data_dir}/annotations")
        print("💡 Make sure you've run 1_extract_frames.py first")
        return
    
    # Start annotation
    annotator = VenturaBeeAnnotator(data_dir)
    annotator.start_annotation()

if __name__ == "__main__":
    main()