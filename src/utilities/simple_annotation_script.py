#!/usr/bin/env python3
"""
Simple annotation script using matplotlib (more reliable than Qt-based tools)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import json
from pathlib import Path

class SimpleAnnotator:
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_files = sorted([f for f in self.image_dir.glob("*.jpg")])  # Show all images
        self.current_idx = 0
        self.boxes = []
        self.fig = None
        self.ax = None
        self.current_image = None
        
        print(f"Found {len(self.image_files)} bee images to annotate")
        print("Bee frames:", [f.name for f in self.image_files])
        
    def load_image(self):
        """Load current image"""
        if self.current_idx >= len(self.image_files):
            print("✅ All images annotated!")
            return False
            
        image_path = self.image_files[self.current_idx]
        self.current_image = cv2.imread(str(image_path))
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Load existing annotations if any
        label_path = self.label_dir / (image_path.stem + ".txt")
        print(f"🔍 DEBUG: Looking for label file: {label_path}")
        print(f"🔍 DEBUG: Label file exists: {label_path.exists()}")
        
        self.boxes = []
        if label_path.exists():
            print(f"✅ DEBUG: Loading annotations from {label_path}")
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        print(f"🔍 DEBUG: Raw annotation line: {parts}")
                        if len(parts) == 5:
                            class_id, cx, cy, w, h = map(float, parts)
                            print(f"🔍 DEBUG: YOLO coords: cx={cx}, cy={cy}, w={w}, h={h}")
                            # Convert from YOLO format to pixel coordinates
                            img_h, img_w = self.current_image.shape[:2]
                            print(f"🔍 DEBUG: Image size: {img_w}x{img_h}")
                            x = (cx - w/2) * img_w
                            y = (cy - h/2) * img_h
                            width = w * img_w
                            height = h * img_h
                            print(f"🔍 DEBUG: Pixel coords: x={x}, y={y}, w={width}, h={height}")
                            self.boxes.append([x, y, width, height])
            print(f"🔍 DEBUG: Total boxes loaded: {len(self.boxes)}")
        
        return True
        
    def display_image(self):
        """Display image with current annotations"""
        if self.fig:
            plt.close(self.fig)
            
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.ax.imshow(self.current_image)
        
        # Draw existing boxes
        for i, (x, y, w, h) in enumerate(self.boxes):
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            self.ax.add_patch(rect)
            self.ax.text(x, y-5, f'Bee {i+1}', color='red', fontweight='bold')
        
        self.ax.set_title(f"Image {self.current_idx + 1}/{len(self.image_files)}: {self.image_files[self.current_idx].name}")
        self.ax.axis('off')
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        # Add control buttons
        ax_next = plt.axes([0.85, 0.01, 0.1, 0.04])
        ax_save = plt.axes([0.74, 0.01, 0.1, 0.04])
        ax_clear = plt.axes([0.63, 0.01, 0.1, 0.04])
        ax_undo = plt.axes([0.52, 0.01, 0.1, 0.04])
        
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_undo = Button(ax_undo, 'Undo')
        
        self.btn_next.on_clicked(self.next_image)
        self.btn_save.on_clicked(self.save_annotations)
        self.btn_clear.on_clicked(self.clear_boxes)
        self.btn_undo.on_clicked(self.undo_last)
        
        self.start_x = None
        self.start_y = None
        
        plt.tight_layout()
        plt.show()
    
    def on_press(self, event):
        """Mouse press event"""
        if event.inaxes != self.ax:
            return
        self.start_x = event.xdata
        self.start_y = event.ydata
        
    def on_release(self, event):
        """Mouse release event - create bounding box"""
        if event.inaxes != self.ax or self.start_x is None:
            return
            
        end_x = event.xdata
        end_y = event.ydata
        
        # Calculate box coordinates
        x = min(self.start_x, end_x)
        y = min(self.start_y, end_y)
        width = abs(end_x - self.start_x)
        height = abs(end_y - self.start_y)
        
        # Only add if box is large enough
        if width > 10 and height > 10:
            self.boxes.append([x, y, width, height])
            print(f"Added bounding box: ({x:.1f}, {y:.1f}) {width:.1f}x{height:.1f}")
            self.display_image()  # Refresh display
    
    def save_annotations(self, event=None):
        """Save current annotations in YOLO format"""
        image_path = self.image_files[self.current_idx]
        label_path = self.label_dir / (image_path.stem + ".txt")
        
        img_h, img_w = self.current_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for x, y, w, h in self.boxes:
                # Convert to YOLO format (normalized)
                center_x = (x + w/2) / img_w
                center_y = (y + h/2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        print(f"✅ Saved {len(self.boxes)} annotations to {label_path}")
    
    def next_image(self, event=None):
        """Move to next image"""
        self.save_annotations()
        self.current_idx += 1
        if self.load_image():
            self.display_image()
        else:
            plt.close('all')
            print("🎉 Annotation complete!")
    
    def clear_boxes(self, event=None):
        """Clear all bounding boxes"""
        self.boxes = []
        self.display_image()
        print("Cleared all boxes")
    
    def undo_last(self, event=None):
        """Remove last bounding box"""
        if self.boxes:
            removed = self.boxes.pop()
            self.display_image()
            print(f"Removed box: {removed}")
    
    def start_annotation(self):
        """Start the annotation process"""
        print(f"\n🎯 STARTING ANNOTATION")
        print(f"{'='*50}")
        print(f"Instructions:")
        print(f"1. Click and drag to draw bounding boxes around bees")
        print(f"2. Click 'Save' to save current annotations")
        print(f"3. Click 'Next →' to go to next image")
        print(f"4. Use 'Undo' to remove last box, 'Clear' to remove all")
        print(f"5. Close window when done")
        print(f"{'='*50}")
        
        if self.load_image():
            self.display_image()
        else:
            print("No images to annotate")


def main():
    """Run the annotation tool"""
    image_dir = "ventura_object_detection_dataset/images/train"
    label_dir = "ventura_object_detection_dataset/labels/train"
    
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    if not os.path.exists(label_dir):
        print(f"❌ Label directory not found: {label_dir}")
        return
    
    annotator = SimpleAnnotator(image_dir, label_dir)
    annotator.start_annotation()

if __name__ == "__main__":
    main()