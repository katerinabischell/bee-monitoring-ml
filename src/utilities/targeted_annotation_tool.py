#!/usr/bin/env python3
"""
Targeted annotation tool that focuses ONLY on the frames where you confirmed seeing bees
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from pathlib import Path

class TargetedBeeAnnotator:
    """
    Annotate ONLY the specific frames where you confirmed seeing bees
    """
    
    def __init__(self):
        # These are the exact timestamps where you confirmed seeing bees in the debug
        self.target_frames = [
            {'timestamp': 2, 'note': '1st bombus on the left'},
            {'timestamp': 3, 'note': '1st bombus on the left'},
            {'timestamp': 4, 'note': '2nd bombus flies in on the left'},
            {'timestamp': 7, 'note': '2nd bombus visible clearly on the left'},
            {'timestamp': 10, 'note': '2nd bombus flies over to 1st bombus'},
            {'timestamp': 15, 'note': '1st and 2nd bombus visible both flying on the left'},
            {'timestamp': 16, 'note': '2nd bombus visible, the 1st left the frame'},
            {'timestamp': 17, 'note': '2nd bombus visible'},
            {'timestamp': 27, 'note': '3rd bombus flies into frame'},
            {'timestamp': 36, 'note': '3rd bombus flies into frame on the left'},
            {'timestamp': 38, 'note': '3rd bombus flying in frame'},
            {'timestamp': 57, 'note': '3rd bombus seen on top flowers (partially obscured)'},
            {'timestamp': 62, 'note': '3rd bombus is seen on top in middle again'},
            {'timestamp': 68, 'note': '3rd bombus seen flying left'},
            {'timestamp': 69, 'note': '3rd bombus lands on flower top left'},
            {'timestamp': 71, 'note': '3rd bombus is seen clearly top left'},
            {'timestamp': 240, 'note': '4th bombus comes into frame from bottom right'},
            {'timestamp': 242, 'note': '4th bombus very visible bottom right'},
            {'timestamp': 292, 'note': '5th bombus visible on top right'},
            {'timestamp': 315, 'note': '6th bombus flies into frame top middle'},
            {'timestamp': 323, 'note': '6th bombus seen clearly top middle'},
            {'timestamp': 325, 'note': '7th bombus comes into frame (both 6 and 7 are visible)'},
            {'timestamp': 326, 'note': '6th and 7th bombus clearly visible top middle - TWO BEES!'},
            {'timestamp': 336, 'note': '6th and 7th bombus clearly visible top middle - TWO BEES!'},
            {'timestamp': 338, 'note': '7th bombus flies out of frame at top, 6th still visible'},
            {'timestamp': 355, 'note': '8th bombus flies into left side, 6th bombus still visible - TWO BEES!'},
            {'timestamp': 357, 'note': '8th still visible on right, 6th gets covered by a flower'},
            {'timestamp': 364, 'note': '8th bombus clearly visible on left'},
            {'timestamp': 369, 'note': '8th bombus moved to top left'},
            {'timestamp': 380, 'note': '8th bombus top left'},
            {'timestamp': 386, 'note': '8th bombus top left'},
            {'timestamp': 420, 'note': '9th bombus flies in top right'},
            {'timestamp': 422, 'note': '9th bombus flies out at top'},
            {'timestamp': 430, 'note': '9th bombus flies back at top'},
            {'timestamp': 433, 'note': '9th bombus seen top middle'},
            {'timestamp': 456, 'note': '10th bombus seen top right corner'},
            {'timestamp': 471, 'note': '11th bombus seen top left corner'},
        ]
        
        self.image_dir = Path("ventura_object_detection_dataset/images/train")
        self.label_dir = Path("ventura_object_detection_dataset/labels/train")
        
        # Find which target frames actually exist
        self.available_frames = []
        self.find_available_frames()
        
        self.current_idx = 0
        self.boxes = []
        self.fig = None
        self.ax = None
        self.current_image = None
    
    def find_available_frames(self):
        """Find which of our target frames actually exist"""
        
        print(f"🔍 LOOKING FOR TARGET BEE FRAMES")
        print(f"{'='*50}")
        print(f"Looking in: {self.image_dir}")
        
        if not self.image_dir.exists():
            print(f"❌ Directory not found: {self.image_dir}")
            return
        
        for target in self.target_frames:
            timestamp = target['timestamp']
            
            # Look for the frame file (various possible formats)
            possible_names = [
                f"P1000446_ventura_t{timestamp:04d}.jpg",
                f"P1000446_ventura_t{timestamp:03d}.jpg",
                f"P1000446_ventura_t{timestamp:02d}.jpg",
            ]
            
            found = False
            for name in possible_names:
                frame_path = self.image_dir / name
                if frame_path.exists():
                    self.available_frames.append({
                        'timestamp': timestamp,
                        'note': target['note'],
                        'path': frame_path,
                        'filename': name
                    })
                    print(f"   ✅ {timestamp:3d}s: {name} - {target['note']}")
                    found = True
                    break
            
            if not found:
                print(f"   ❌ {timestamp:3d}s: Not found - {target['note']}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Target frames: {len(self.target_frames)}")
        print(f"   Available frames: {len(self.available_frames)}")
        
        if len(self.available_frames) == 0:
            print("\n❌ NO TARGET FRAMES FOUND!")
            print("💡 This means the frame extraction didn't create the expected files")
            print("Available files:")
            for file in sorted(self.image_dir.glob("*.jpg"))[:10]:
                print(f"      {file.name}")
        else:
            print(f"\n🎯 KEY FRAMES TO FOCUS ON:")
            priority_frames = [f for f in self.available_frames if 'TWO BEES' in f['note']]
            for frame in priority_frames:
                print(f"   {frame['timestamp']:3d}s: {frame['note']}")
    
    def start_annotation(self):
        """Start annotating the available bee frames"""
        
        if len(self.available_frames) == 0:
            print("❌ No frames to annotate")
            return
        
        print(f"\n🎯 STARTING TARGETED ANNOTATION")
        print(f"{'='*50}")
        print(f"Annotating {len(self.available_frames)} confirmed bee frames")
        print(f"Instructions:")
        print(f"1. Draw tight boxes around each visible bee")
        print(f"2. For frames with 'TWO BEES', draw 2 separate boxes")
        print(f"3. Use buttons to save and navigate")
        print(f"{'='*50}")
        
        self.load_current_frame()
        self.display_frame()
    
    def load_current_frame(self):
        """Load the current frame"""
        if self.current_idx >= len(self.available_frames):
            print("🎉 All target frames annotated!")
            return False
        
        frame_info = self.available_frames[self.current_idx]
        self.current_image = cv2.imread(str(frame_info['path']))
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Load existing annotations
        timestamp = frame_info['timestamp']
        label_file = self.label_dir / f"P1000446_ventura_t{timestamp:04d}.txt"
        
        self.boxes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, cx, cy, w, h = map(float, parts)
                            # Convert from YOLO format to pixel coordinates
                            img_h, img_w = self.current_image.shape[:2]
                            x = (cx - w/2) * img_w
                            y = (cy - h/2) * img_h
                            width = w * img_w
                            height = h * img_h
                            self.boxes.append([x, y, width, height])
        
        return True
    
    def display_frame(self):
        """Display current frame with annotation interface"""
        if self.current_idx >= len(self.available_frames):
            return
        
        frame_info = self.available_frames[self.current_idx]
        
        if self.fig:
            plt.close(self.fig)
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        self.ax.imshow(self.current_image)
        
        # Draw existing boxes
        for i, (x, y, w, h) in enumerate(self.boxes):
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            self.ax.add_patch(rect)
            self.ax.text(x, y-5, f'Bee {i+1}', color='red', fontweight='bold', fontsize=12)
        
        # Title with frame info
        title = f"Frame {self.current_idx + 1}/{len(self.available_frames)}: {frame_info['timestamp']}s\n{frame_info['note']}"
        self.ax.set_title(title, fontsize=14, pad=20)
        self.ax.axis('off')
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        # Control buttons
        ax_next = plt.axes([0.85, 0.01, 0.1, 0.04])
        ax_save = plt.axes([0.74, 0.01, 0.1, 0.04])
        ax_clear = plt.axes([0.63, 0.01, 0.1, 0.04])
        ax_skip = plt.axes([0.52, 0.01, 0.1, 0.04])
        
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_skip = Button(ax_skip, 'Skip')
        
        self.btn_next.on_clicked(self.next_frame)
        self.btn_save.on_clicked(self.save_annotations)
        self.btn_clear.on_clicked(self.clear_boxes)
        self.btn_skip.on_clicked(self.skip_frame)
        
        self.start_x = None
        self.start_y = None
        
        plt.tight_layout()
        plt.show()
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.start_x = event.xdata
        self.start_y = event.ydata
    
    def on_release(self, event):
        if event.inaxes != self.ax or self.start_x is None:
            return
        
        end_x = event.xdata
        end_y = event.ydata
        
        x = min(self.start_x, end_x)
        y = min(self.start_y, end_y)
        width = abs(end_x - self.start_x)
        height = abs(end_y - self.start_y)
        
        if width > 10 and height > 10:
            self.boxes.append([x, y, width, height])
            print(f"Added bee #{len(self.boxes)}: ({x:.1f}, {y:.1f}) {width:.1f}x{height:.1f}")
            self.display_frame()
    
    def save_annotations(self, event=None):
        """Save annotations in YOLO format"""
        if self.current_idx >= len(self.available_frames):
            return
        
        frame_info = self.available_frames[self.current_idx]
        timestamp = frame_info['timestamp']
        label_file = self.label_dir / f"P1000446_ventura_t{timestamp:04d}.txt"
        
        img_h, img_w = self.current_image.shape[:2]
        
        with open(label_file, 'w') as f:
            for x, y, w, h in self.boxes:
                center_x = (x + w/2) / img_w
                center_y = (y + h/2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        print(f"✅ Saved {len(self.boxes)} bee annotations for {timestamp}s")
    
    def next_frame(self, event=None):
        """Move to next frame"""
        self.save_annotations()
        self.current_idx += 1
        if self.load_current_frame():
            self.display_frame()
        else:
            plt.close('all')
            print("🎉 All bee frames annotated!")
    
    def skip_frame(self, event=None):
        """Skip current frame without saving"""
        self.current_idx += 1
        if self.load_current_frame():
            self.display_frame()
        else:
            plt.close('all')
            print("🎉 Finished reviewing frames!")
    
    def clear_boxes(self, event=None):
        """Clear all boxes"""
        self.boxes = []
        self.display_frame()

def main():
    """Run targeted annotation"""
    annotator = TargetedBeeAnnotator()
    annotator.start_annotation()

if __name__ == "__main__":
    main()