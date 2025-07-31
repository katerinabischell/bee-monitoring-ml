# Bombus Detection for Endangered Plant Conservation

**AI-powered pollinator monitoring system for Salt Marsh Bird's Beak and Ventura Marsh Milkvetch conservation**

*Author: Katerina Bischel*  
*Institution: Bren School of Environmental Science & Management, UCSB*  
*Project: Cheadle Center for Biodiversity and Ecological Restoration*
*Advisors: Cheadle Center for Biodiversity and Ecological Restoration - Chris Evelyn and Katja Seltmann*
*Supported by: National Science Foundation Project Extending Anthophila research through image and trait digitalization (Big-Bee #DBI2102006)*

---

## Project Overview

This project develops machine learning tools to automatically detect *Bombus vosnesenskii* (Yellow-faced Bumble Bee) pollinators in camera trap footage from endangered coastal plant restoration sites. The system supports conservation efforts for two federally endangered species:

- **Salt Marsh Bird's Beak** (*Chloropyron maritimum* ssp. *maritimum*)
- **Ventura Marsh Milkvetch** (*Astragalus pycnostachyus* var. *lanosissimus*)

### Conservation Context

Salt Marsh Bird's Beak populations at North Campus Open Space (NCOS) have grown from 57 individuals (2023) to 15,542 (2025) through systematic restoration efforts. This project provides the technology to monitor pollinator services critical for reproduction and long-term species recovery.

---

## Key Results

### Model Performance
- **100% Test Accuracy** on held-out test frames (89 frames)
- **Perfect Precision & Recall** (1.000) on laboratory test set
- **94.1% Field Validation Accuracy** on 152 real-world observations
- **93.1% Recall** and **79.4% Precision** in field deployment conditions

### Dataset Characteristics
- **42 videos processed** from weeks 2-4 of 2025 field season
- **1,036 frames extracted** automatically from camera trap footage
- **361 positive frames** with confirmed bombus activity
- **675 negative frames** without bombus activity

### Real-World Validation
**Correctly identified videos with NO bombus activity** (0% detection rate)  
**Correctly identified videos with EXTENSIVE bombus activity** (100% detection rate, perfect confidence)

---

## 🔧 Technical Architecture

### Model Design
- **Base Architecture**: ResNet-18 with transfer learning
- **Input**: 224×224 RGB frames extracted from video
- **Output**: Binary classification (Bombus Present/Absent)
- **Training**: Custom CNN with data augmentation and class weighting

### Data Pipeline
1. **Video Processing**: Automated frame extraction from MP4 files
2. **Observation Integration**: Field notes automatically label training data
3. **Data Augmentation**: Handles class imbalance and improves generalization
4. **Model Training**: Transfer learning with early stopping and validation

---

## Repository Structure

```
bee-monitoring-ml/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Collection Observations3.xlsx     # Field observation data
│
├── src/
│   ├── preliminary_analysis_00.py    # Initial dataset analysis
│   ├── video_processing_pipeline.py  # Frame extraction & labeling
│   ├── bombus_model_trainer.py       # Model training pipeline
│   └── correct_video_analyzer.py     # Video analysis with trained model
│
├── data/
│   ├── processed/
│   │   ├── frames/
│   │   │   ├── positive/              # Frames with bombus activity
│   │   │   └── negative/              # Frames without bombus activity
│   │   └── annotations/
│   │       ├── frame_annotations.csv # Frame-level metadata
│   │       └── processing_summary.json
│   └── raw/                           # Original video files (external drive)
│
├── models/
│   ├── best_bombus_model.pth         # Trained model weights
│   ├── training_results.json         # Training metrics & parameters
│   └── training_history.png          # Loss/accuracy curves
│
└── analysis_output/
    ├── analysis_summary_report.md    # Dataset analysis report
    ├── pollinator_analysis_summary.png
    └── detailed_analysis.json
```

---

## Quick Start

### Prerequisites
```bash
pip install torch torchvision opencv-python pandas numpy matplotlib seaborn openpyxl pillow scikit-learn
```

### 1. Analyze Your Dataset
```bash
python3 preliminary_analysis_00.py
```
**Output**: Dataset statistics, bombus observation counts, ML recommendations

### 2. Process Videos & Extract Frames
```bash
python3 video_processing_pipeline.py
```
**Input**: Path to video directory  
**Output**: Labeled training frames in `data/processed/frames/`

### 3. Train the Model
```bash
python3 bombus_model_trainer.py
```
**Output**: Trained model (`best_bombus_model.pth`), training metrics, performance plots

### 4. Analyze New Videos
```bash
python3 correct_video_analyzer.py "/path/to/video.mp4"
```
**Output**: Detection timeline, confidence scores, activity summary

### Updates for synthetic training data
- down_sample_video.py: downsamples video to half the size
- spit_video_frames.py: splits video into single frames
- render_bee_views.py: uses 3D photogrammetry model to create multiple views of the bee. Currently there is an issue with a shadow from a lighting source that should be fixed. Requires Blender.
- autocrop_bee.py: Crops the output from render_bee_views.py to just pixels
- overlay_bees_random_scaled_rotated_limit10.py: creates a transparent overlay of the bee on the video images.

  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16415880.svg)](https://doi.org/10.5281/zenodo.16415880)

---

## Dataset Details

### Field Data Collection
- **Location**: North Campus Open Space (NCOS), UC Santa Barbara
- **Species**: Salt Marsh Bird's Beak (*Chloropyron maritimum*)
- **Cameras**: GoPro camera traps at multiple sites
- **Schedule**: Morning, midday, and afternoon shifts
- **Duration**: 21-30 minute recordings per session

### Observation Protocol
- **Systematic field notes** for each video recording
- **Species-level identification** (*B. vosnesenskii* vs *B. californicus*)
- **Behavioral observations** (on plant vs around plant)
- **Environmental conditions** (weather, wind, temperature)

### Video Specifications
- **Format**: MP4 (H.264 encoding)
- **Resolution**: 1920×1080 pixels
- **Frame Rate**: ~60 FPS
- **Duration**: 1200-1800 seconds per video
- **Storage**: External drive due to file sizes (~1GB per video)

---

## Methodology

### Computer Vision Pipeline

1. **Frame Extraction**
   - Extract 25 frames per video using uniform sampling
   - Resize to 224×224 pixels for CNN input
   - Convert to RGB and normalize pixel values

2. **Automatic Labeling**
   - Parse video metadata (week, site, camera, shift)
   - Match with field observation records
   - Label frames based on confirmed bombus activity

3. **Data Augmentation**
   - Random horizontal flips (50% probability)
   - Random rotation (±15 degrees)
   - Color jittering (brightness, contrast, saturation)
   - Random resized cropping (80-100% scale)

4. **Model Training**
   - Transfer learning with pre-trained ResNet-18
   - Class-weighted loss function for imbalanced data
   - Adam optimizer with learning rate scheduling
   - Early stopping based on validation accuracy

### Validation Strategy
- **70/15/15 split** for train/validation/test
- **Stratified sampling** maintains class balance
- **Hold-out test set** for unbiased performance evaluation
- **Real-world validation** on original video files

---

## Results & Impact

### Model Performance Metrics
```
Classification Report:
                precision    recall  f1-score   support
     No Bombus       1.00      1.00      1.00        58
Bombus Present       1.00      1.00      1.00        31
      accuracy                           1.00        89

Confusion Matrix:
              Predicted
Actual    No Bombus  Bombus
No Bombus      58      0
Bombus          0     31
```

### Conservation Applications

**Immediate Benefits:**
- **Automated video analysis** eliminates manual review of 100+ hours of footage
- **Quantitative pollinator metrics** support adaptive habitat management
- **Real-time deployment** enables responsive conservation actions

**Long-term Impact:**
- **Scalable monitoring** across multiple restoration sites
- **Temporal pattern analysis** reveals optimal pollinator activity windows
- **Evidence-based management** decisions for endangered species recovery

---

## Advanced Usage

### Batch Video Processing
```python
from video_processing_pipeline import VideoProcessor

processor = VideoProcessor(
    video_root_dir="/path/to/videos",
    observations_file="Collection Observations3.xlsx"
)
processor.process_all_videos()
```

### Custom Model Training
```python
from bombus_model_trainer import BombusClassifier, BombusTrainer

model = BombusClassifier(num_classes=2)
trainer = BombusTrainer(model)
trainer.train(train_loader, val_loader, epochs=50)
```

### Video Analysis API
```python
from correct_video_analyzer import VideoAnalyzer

analyzer = VideoAnalyzer(model_path='best_bombus_model.pth')
results = analyzer.analyze_video('/path/to/video.mp4')
```

---

## Future Developments

### Short-term Enhancements
- [ ] **Multi-species classification** (*B. vosnesenskii* vs *B. californicus*)
- [ ] **Object detection** with bounding boxes around pollinators
- [ ] **Temporal modeling** using video sequences instead of single frames
- [ ] **Real-time processing** for live camera feeds

### Research Extensions
- [ ] **Behavioral analysis** (foraging patterns, visit duration)
- [ ] **Pollination effectiveness** metrics
- [ ] **Cross-site deployment** to other restoration projects
- [ ] **Climate correlation** analysis with pollinator activity

### Technology Integration
- [ ] **Mobile app** for field deployment
- [ ] **Cloud processing** pipeline for large datasets
- [ ] **Integration** with existing camera trap networks
- [ ] **Dashboard** for real-time monitoring

---

## Scientific Background

### Target Species

**Salt Marsh Bird's Beak** (*Chloropyron maritimum* ssp. *maritimum*)
- **Status**: Federally endangered, state endangered
- **Habitat**: Coastal salt marsh, sandy soils near slough edges
- **Pollination**: Depends on native bee species for reproduction
- **Threats**: Habitat loss, invasive species, climate change

**Bombus vosnesenskii** (Yellow-faced Bumble Bee)
- **Role**: Primary pollinator for endangered coastal plants
- **Foraging**: Generalist species, active throughout growing season
- **Conservation**: Critical for plant species recovery efforts

### Ecological Context
The restoration success at NCOS provides a unique opportunity to study plant-pollinator interactions in recovering coastal ecosystems. Understanding pollinator visitation patterns is essential for:

- **Habitat optimization** (plant density, spatial configuration)
- **Pollinator corridor** design and management
- **Adaptive management** responses to changing conditions
- **Long-term monitoring** of ecosystem recovery

---

## Contact & Collaboration

**Katerina Bischel**  
Graduate Student, Bren School of Environmental Science & Management  
University of California, Santa Barbara  

**Project Partners:**
- **Cheadle Center for Biodiversity and Ecological Restoration** - Chris Evelyn and Katja Seltmann


**For questions about:**
- **Research methodology**: Contact project team
- **Data access**: See institutional data sharing policies
- **Collaboration opportunities**: Open to partnerships with conservation organizations
- **Technical implementation**: See Issues tab for bug reports and feature requests

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{bischel2025bombus,
  title={Bombus Detection for Endangered Plant Conservation: AI-powered Pollinator Monitoring},
  author={Bischel, Katerina},
  year={2025},
  institution={Bren School of Environmental Science \& Management, UC Santa Barbara},
  url={https://github.com/[katerinabischell]/bee-monitoring-ml}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Video data and field observations remain proprietary to UC Santa Barbara and research partners. The code and methodologies are open source for conservation applications.

---

## Acknowledgments

- **UC Santa Barbara** Cheadle Center for Biodiversity and Ecological Restoration - Chris Evelyn and Katja Seltmann
- **National Science Foundation** project Extending Anthophila research through image and trait digitalization (Big-Bee #DBI2102006)
- **NCOS restoration team** for maintaining field sites and equipment
- **PyTorch community** for deep learning framework
- **OpenCV contributors** for computer vision tools

---

*This project demonstrates how AI technology can directly support endangered species conservation through automated, scalable monitoring systems. The success of this bombus detection system provides a template for applying machine learning to other wildlife monitoring challenges in conservation biology.*

---

**Project Status**: **Production Ready** - Model validated and deployment ready  
**Last Updated**: July 2025  
**Version**: 1.0.0
