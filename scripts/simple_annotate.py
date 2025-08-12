#!/usr/bin/env python3
"""
Simple, stable matplotlib annotation tool
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import pandas as pd

class SimpleStableAnnotator:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "extracted_frames"
        self.label_dir = self.data_dir / "annotations"
        
        # Get all JPG files and sort them
        self.image_files = sorted([f for f in self.image_dir.glob("*.jpg")])
        
        # Find where we left off
        self.current_idx = self.find_last_annotated_frame()
        self.boxes = []
        self.current_image = None
        
        # Load annotation summary
        self.load_annotation_summary()
        
        print(f"🎯 SIMPLE BEE ANNOTATOR")
        print(f"Found {len(self.image_files)} frames")
        print(f"Starting from frame {self.current_idx + 1}")
        
    def find_last_annotated_frame(self):
        """Find the last manually annotated frame"""
        import time
        cutoff_time = time.time() - (24 * 3600)  # Files modified in last 24 hours
        
        last_idx = 0
        for i, img_file in enumerate(self.image_files):
            label_file = self.label_dir / (img_file.stem + ".txt")
            if label_file.exists():
                if label_file.stat().st_mtime > cutoff_time:
                    last_idx = i
        
        return min(last_idx + 1, len(self.image_files) - 1)
    
    def load_annotation_summary(self):
        """Load annotation summary"""
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
            return f"Time: {info['timestamp']} | Expected: {info['bee_count']} bees | Quadrants: {info['quadrants']}"
        return "No info"
    
    def load_image_and_annotations(self):
        """Load current image and its annotations"""
        if self.current_idx >= len(self.image_files):
            return False
            
        image_path = self.image_files[self.current_idx]
        self.current_image = cv2.imread(str(image_path))
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Load existing annotations
        label_path = self.label_dir / (image_path.stem + ".txt")
        self.boxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            parts = line.split()
                            if len(parts) == 5:
                                _, cx, cy, w, h = map(float, parts)
                                # Convert to pixel coordinates
                                img_h, img_w = self.current_image.shape[:2]
                                x = (cx - w/2) * img_w
                                y = (cy - h/2) * img_h
                                width = w * img_w
                                height = h * img_h
                                self.boxes.append([x, y, width, height])
                        except:
                            pass
        
        print(f"📷 Frame {self.current_idx + 1}: {image_path.name}")
        print(f"   {self.get_frame_info(image_path.name)}")
        print(f"   Loaded {len(self.boxes)} existing boxes")
        return True
    
    def show_image(self):
        """Show image with current annotations"""
        plt.close('all')  # Close any existing windows
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.current_image)
        
        # Draw boxes
        for i, (x, y, w, h) in enumerate(self.boxes):
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-10, f'Bee {i+1}', color='red', fontweight='bold', fontsize=12)
        
        filename = self.image_files[self.current_idx].name
        ax.set_title(f"Frame {self.current_idx + 1}/{len(self.image_files)}: {filename}\n{self.get_frame_info(filename)}", 
                     fontsize=10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def add_box_interactive(self):
        """Add a bounding box interactively"""
        print("Click two points to define bounding box (top-left, bottom-right)")
        
        fig, ax = self.show_image()
        points = []
        
        def onclick(event):
            if event.inaxes == ax:
                points.append([event.xdata, event.ydata])
                ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
                fig.canvas.draw()
                
                if len(points) == 2:
                    # Calculate box
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    
                    if w > 5 and h > 5:
                        self.boxes.append([x, y, w, h])
                        print(f"✅ Added box {len(self.boxes)}: ({x:.1f}, {y:.1f}) {w:.1f}x{h:.1f}")
                        
                        # Draw new box
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x, y-10, f'Bee {len(self.boxes)}', color='lime', fontweight='bold', fontsize=12)
                        fig.canvas.draw()
                    
                    points.clear()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    
    def save_annotations(self):
        """Save current annotations"""
        image_path = self.image_files[self.current_idx]
        label_path = self.label_dir / (image_path.stem + ".txt")
        
        img_h, img_w = self.current_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for x, y, w, h in self.boxes:
                # Convert to YOLO format
                center_x = (x + w/2) / img_w
                center_y = (y + h/2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # Clamp to valid range
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))
                
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        print(f"💾 Saved {len(self.boxes)} annotations")
    
    def run_interactive(self):
        """Run the interactive annotation session"""
        print(f"\n🎯 INTERACTIVE ANNOTATION MODE")
        print(f"Commands:")
        print(f"  'n' or Enter: Next frame")
        print(f"  'p': Previous frame") 
        print(f"  'a': Add bounding box")
        print(f"  'c': Clear all boxes")
        print(f"  'u': Undo last box")
        print(f"  's': Save annotations")
        print(f"  'j': Jump to frame")
        print(f"  'q': Quit")
        print(f"  'h': Help")
        
        while True:
            if not self.load_image_and_annotations():
                print("🎉 All frames complete!")
                break
            
            self.show_image()
            
            try:
                cmd = input(f"\nFrame {self.current_idx + 1}/{len(self.image_files)} > ").strip().lower()
                
                if cmd in ['n', '']:
                    self.save_annotations()
                    self.current_idx += 1
                elif cmd == 'p':
                    if self.current_idx > 0:
                        self.save_annotations()
                        self.current_idx -= 1
                    else:
                        print("Already at first frame")
                elif cmd == 'a':
                    self.add_box_interactive()
                elif cmd == 'c':
                    self.boxes = []
                    print("🗑️ Cleared all boxes")
                elif cmd == 'u':
                    if self.boxes:
                        removed = self.boxes.pop()
                        print(f"↶ Removed box: {removed}")
                    else:
                        print("No boxes to remove")
                elif cmd == 's':
                    self.save_annotations()
                elif cmd == 'j':
                    try:
                        frame_num = int(input("Jump to frame: "))
                        if 1 <= frame_num <= len(self.image_files):
                            self.save_annotations()
                            self.current_idx = frame_num - 1
                        else:
                            print(f"Invalid frame. Must be 1-{len(self.image_files)}")
                    except ValueError:
                        print("Invalid number")
                elif cmd == 'q':
                    self.save_annotations()
                    print("👋 Goodbye!")
                    break
                elif cmd == 'h':
                    print("Commands: n=next, p=prev, a=add box, c=clear, u=undo, s=save, j=jump, q=quit")
                else:
                    print("Unknown command. Type 'h' for help")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    annotator = SimpleStableAnnotator()
    annotator.run_interactive()

if __name__ == "__main__":
    main()