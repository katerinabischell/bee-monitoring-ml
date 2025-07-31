
# Bounding Box Annotation Guide

## Tools Needed
1. **LabelImg**: pip install labelimg
2. **CVAT**: Online annotation tool
3. **Roboflow**: Web-based annotation

## YOLO Format
Format: class_id center_x center_y width height
- All values normalized 0-1
- center_x, center_y: center of bounding box
- width, height: box dimensions

## Example for bee at center of image:
0 0.5 0.5 0.2 0.3

## Annotation Guidelines
1. **Draw tight boxes** around visible bees
2. **Include partial bees** if >50% visible
3. **Use class 0** for all bombus species
4. **Be consistent** with box sizing

## Quick Start with LabelImg:
```bash
pip install labelimg
labelimg object_detection_dataset/images/train object_detection_dataset/labels/train
```

Navigate through images and draw bounding boxes around bees.
