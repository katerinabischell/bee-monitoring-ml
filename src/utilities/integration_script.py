#!/usr/bin/env python3
"""
Integration script for upgrading current bombus detection to object detection and species classification
Compatible with existing bee-monitoring-ml repository structure
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

# Add your existing src directory to path
sys.path.append('src')

# Import your existing modules (adapt these imports based on your actual module names)
try:
    from correct_video_analyzer import VideoAnalyzer  # Your current binary classifier
except ImportError:
    print("⚠️ Could not import existing VideoAnalyzer. Please ensure src/ directory is accessible.")

class WorkflowUpgrader:
    """
    Upgrade your existing binary classification workflow to support:
    1. Object detection with bee counting
    2. Species classification
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.setup_directories()
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        default_config = {
            "models": {
                "binary_classifier": "models/best_bombus_model.pth",
                "object_detector": "models/bombus_object_detection_model.pth", 
                "species_classifier": "models/bombus_species_classifier.pth"
            },
            "video_processing": {
                "segment_duration": 300,  # 5 minutes
                "sample_interval": 30,    # Sample every 30 seconds for species classification
                "confidence_threshold": 0.7
            },
            "output_directories": {
                "binary_analysis": "analysis_output/",
                "object_detection": "object_detection_results/",
                "species_classification": "species_analysis_results/",
                "combined_reports": "combined_analysis_reports/"
            },
            "field_data": {
                "observations_file": "Collection Observations3.xlsx",
                "validation_results": "field_ai_validation/detailed_validation_results.csv"
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge with defaults
            for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value
            return default_config
        else:
            # Create default config file
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"📄 Created default config file: {config_path}")
            return default_config
    
    def setup_directories(self):
        """Create output directories"""
        for dir_name in self.config["output_directories"].values():
            os.makedirs(dir_name, exist_ok=True)
    
    def analyze_current_model_performance(self):
        """
        Analyze your current model's performance to understand what it's actually doing
        """
        print("🔍 Analyzing current model performance...")
        
        # Load your existing validation results
        if os.path.exists(self.config["field_data"]["validation_results"]):
            validation_df = pd.read_csv(self.config["field_data"]["validation_results"])
            
            print(f"📊 Current Model Performance Analysis:")
            print(f"   Total observations: {len(validation_df)}")
            print(f"   Field detection rate: {validation_df['field_bombus_any'].mean()*100:.1f}%")
            print(f"   AI detection rate: {validation_df['ai_prediction'].mean()*100:.1f}%")
            
            # Analyze detection patterns
            true_positives = ((validation_df['field_bombus_any'] == 1) & (validation_df['ai_prediction'] == 1)).sum()
            false_positives = ((validation_df['field_bombus_any'] == 0) & (validation_df['ai_prediction'] == 1)).sum()
            false_negatives = ((validation_df['field_bombus_any'] == 1) & (validation_df['ai_prediction'] == 0)).sum()
            true_negatives = ((validation_df['field_bombus_any'] == 0) & (validation_df['ai_prediction'] == 0)).sum()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   True Positives: {true_positives}")
            print(f"   False Positives: {false_positives}")
            print(f"   False Negatives: {false_negatives}")
            
            # Identify problematic videos for targeted improvement
            problem_videos = validation_df[
                (validation_df['field_bombus_any'] != validation_df['ai_prediction'])
            ]['video_id'].unique()
            
            print(f"   Videos with misclassifications: {len(problem_videos)}")
            
            return {
                'validation_df': validation_df,
                'metrics': {
                    'precision': precision,
                    'recall': recall,
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives
                },
                'problem_videos': problem_videos
            }
        else:
            print("⚠️ Validation results file not found. Skipping performance analysis.")
            return None
    
    def upgrade_to_object_detection(self, video_path: str, output_dir: str = None):
        """
        Upgrade analysis from binary classification to object detection
        """
        if output_dir is None:
            output_dir = self.config["output_directories"]["object_detection"]
        
        print(f"🔧 Upgrading to object detection analysis: {os.path.basename(video_path)}")
        
        # Check if object detection model exists
        if not os.path.exists(self.config["models"]["object_detector"]):
            print("⚠️ Object detection model not found. You'll need to train it first.")
            print("   Recommended: Convert your current binary classifier to object detection format")
            return self.simulate_object_detection_upgrade(video_path, output_dir)
        
        # Import object detection analyzer (from the previous artifact)
        # detector = BombusObjectDetector(self.config["models"]["object_detector"])
        # results = detector.analyze_video_segments(video_path, self.config["video_processing"]["segment_duration"])
        
        # For now, simulate the upgrade
        return self.simulate_object_detection_upgrade(video_path, output_dir)
    
    def simulate_object_detection_upgrade(self, video_path: str, output_dir: str):
        """
        Simulate object detection results based on your current binary classifier
        This bridges the gap until you train an actual object detection model
        """
        print("🔄 Simulating object detection upgrade using current binary classifier...")
        
        # Use your existing video analyzer
        try:
            analyzer = VideoAnalyzer(self.config["models"]["binary_classifier"])
            
            # Run your current analysis
            current_results = analyzer.analyze_video(video_path)
            
            # Convert to object detection format
            simulated_results = self.convert_binary_to_object_detection_format(current_results, video_path)
            
            # Save results
            output_file = os.path.join(output_dir, f"{Path(video_path).stem}_object_detection_simulation.json")
            with open(output_file, 'w') as f:
                json.dump(simulated_results, f, indent=2, default=str)
            
            print(f"✅ Simulated object detection results saved to: {output_file}")
            return simulated_results
            
        except Exception as e:
            print(f"❌ Error running current analyzer: {e}")
            return None
    
    def convert_binary_to_object_detection_format(self, binary_results: dict, video_path: str) -> dict:
        """
        Convert your current binary classification results to object detection format
        """
        # Extract information from your current results
        detection_times = binary_results.get('detection_times', [])
        average_confidence = binary_results.get('average_confidence', 0.0)
        detection_rate = binary_results.get('detection_rate', 0.0)
        
        # Create segments based on detection times
        segments = []
        segment_duration = self.config["video_processing"]["segment_duration"]
        
        # Group detections into 5-minute segments
        current_segment = 0
        segment_start = 0
        
        for detection_time_str in detection_times:
            # Parse time (format: "123.4s")
            detection_time = float(detection_time_str.replace('s', ''))
            
            # Determine which segment this detection belongs to
            segment_num = int(detection_time // segment_duration)
            
            # Create segments up to this detection
            while current_segment <= segment_num:
                segment_end = min((current_segment + 1) * segment_duration, detection_time + segment_duration)
                
                # Count detections in this segment
                segment_detections = [
                    t for t in detection_times 
                    if current_segment * segment_duration <= float(t.replace('s', '')) < segment_end
                ]
                
                segments.append({
                    'segment_number': current_segment + 1,
                    'start_time': current_segment * segment_duration,
                    'end_time': segment_end,
                    'total_bee_detections': len(segment_detections),
                    'max_simultaneous_bees': 1 if len(segment_detections) > 0 else 0,  # Binary classifier assumption
                    'detection_rate': len(segment_detections) / (segment_duration / 30) if segment_duration > 0 else 0,  # Assuming 30s sampling
                    'avg_confidence': average_confidence,
                    'bee_activity_score': len(segment_detections) * average_confidence
                })
                
                current_segment += 1
        
        return {
            'video_path': video_path,
            'analysis_type': 'simulated_object_detection',
            'total_duration': max([float(t.replace('s', '')) for t in detection_times]) if detection_times else 0,
            'segments': segments,
            'summary': {
                'total_segments': len(segments),
                'segments_with_activity': sum(1 for s in segments if s['total_bee_detections'] > 0),
                'total_detections': sum(s['total_bee_detections'] for s in segments),
                'peak_activity_segment': max(segments, key=lambda x: x['bee_activity_score']) if segments else None
            },
            'original_binary_results': binary_results,
            'upgrade_notes': "Simulated from binary classifier. Train object detection model for accurate bee counting."
        }
    
    def create_training_recommendations(self, performance_analysis: dict = None):
        """
        Create recommendations for model improvements based on current performance
        """
        recommendations = {
            'immediate_actions': [],
            'model_improvements': [],
            'data_collection': [],
            'validation_suggestions': []
        }
        
        print("📋 Generating training and improvement recommendations...")
        
        # Based on your meeting notes and current performance
        recommendations['immediate_actions'] = [
            "Break videos into 5-minute segments for more granular analysis",
            "Implement object detection to count multiple bees per frame",
            "Add species classification for B. vosnesenskii, B. californicus, and B. crotchii",
            "Create synthetic dataset using OpenCV overlays with resizing"
        ]
        
        recommendations['model_improvements'] = [
            "Upgrade from ResNet-18 to ResNet-50 or VGG16 for better feature extraction",
            "Implement Faster R-CNN for object detection instead of binary classification",
            "Train multi-class classifier for species identification",
            "Add temporal modeling to track bee movements across frames"
        ]
        
        recommendations['data_collection'] = [
            "Focus on Santa Barbara area bee pictures for species-specific training",
            "Collect more examples of B. californicus (no yellow face/head)",
            "Get examples of B. crotchii (round body, yellow band high on abdomen)",
            "Document bee behavior patterns (on plant vs around plant)"
        ]
        
        if performance_analysis:
            metrics = performance_analysis['metrics']
            
            if metrics['precision'] < 0.85:
                recommendations['model_improvements'].append(
                    f"Reduce false positives (current precision: {metrics['precision']:.3f})"
                )
            
            if metrics['recall'] < 0.90:
                recommendations['model_improvements'].append(
                    f"Reduce false negatives (current recall: {metrics['recall']:.3f})"
                )
            
            if len(performance_analysis['problem_videos']) > 0:
                recommendations['validation_suggestions'].append(
                    f"Review {len(performance_analysis['problem_videos'])} videos with misclassifications"
                )
        
        recommendations['validation_suggestions'].extend([
            "Implement cross-validation with field observations",
            "Create confidence thresholds for different use cases",
            "Validate against manual bee counts in sample videos",
            "Test model performance across different weather conditions"
        ])
        
        # Save recommendations
        output_path = os.path.join(self.config["output_directories"]["combined_reports"], "improvement_recommendations.json")
        with open(output_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print key recommendations
        print("\n🎯 Key Recommendations:")
        for i, action in enumerate(recommendations['immediate_actions'][:3], 1):
            print(f"   {i}. {action}")
        
        return recommendations
    
    def generate_migration_plan(self):
        """
        Generate step-by-step migration plan from current to upgraded system
        """
        migration_plan = {
            'phase_1_preparation': [
                "Backup current models and results",
                "Set up new directory structure for object detection and species classification",
                "Install additional dependencies (YOLO, Detectron2, etc.)",
                "Analyze current model performance and identify improvement areas"
            ],
            'phase_2_object_detection': [
                "Convert existing labeled frames to object detection format (with bounding boxes)",
                "Train Faster R-CNN or YOLO model for bee detection and counting",
                "Validate object detection model against current binary classifier",
                "Implement video segmentation analysis (5-minute chunks)"
            ],
            'phase_3_species_classification': [
                "Collect species-specific training data (synthetic + real)",
                "Train multi-class classifier for 3 bombus species",
                "Integrate species classification with object detection pipeline",
                "Validate species predictions against field observations"
            ],
            'phase_4_integration': [
                "Combine object detection and species classification into unified pipeline",
                "Create comprehensive reporting system",
                "Validate entire system against field data",
                "Deploy for production use on NCOS restoration monitoring"
            ],
            'phase_5_optimization': [
                "Optimize models for real-time processing",
                "Implement automated quality control",
                "Create web dashboard for monitoring results",
                "Scale to other restoration sites"
            ]
        }
        
        # Save migration plan
        output_path = os.path.join(self.config["output_directories"]["combined_reports"], "migration_plan.json")
        with open(output_path, 'w') as f:
            json.dump(migration_plan, f, indent=2)
        
        print("🗺️ Migration plan created. Key phases:")
        for phase, tasks in list(migration_plan.items())[:3]:
            print(f"\n{phase.replace('_', ' ').title()}:")
            for task in tasks[:2]:
                print(f"   • {task}")
        
        return migration_plan


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Upgrade bombus detection workflow")
    parser.add_argument('--mode', choices=['analyze', 'upgrade', 'recommend', 'migrate'], 
                       default='analyze', help='Operation mode')
    parser.add_argument('--video', type=str, help='Video file to analyze')
    parser.add_argument('--video-dir', type=str, help='Directory of videos to process')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize workflow upgrader
    upgrader = WorkflowUpgrader(args.config)
    
    print(f"🚀 Starting workflow upgrade in {args.mode} mode")
    
    if args.mode == 'analyze':
        # Analyze current model performance
        print("\n" + "="*60)
        print("ANALYZING CURRENT MODEL PERFORMANCE")
        print("="*60)
        
        performance = upgrader.analyze_current_model_performance()
        
        if args.video:
            # Analyze specific video with current method
            print(f"\n🎬 Analyzing video: {args.video}")
            try:
                analyzer = VideoAnalyzer(upgrader.config["models"]["binary_classifier"])
                results = analyzer.analyze_video(args.video)
                print(f"✅ Current analysis complete. Detections: {results.get('bombus_detections', 0)}")
            except Exception as e:
                print(f"❌ Error analyzing video: {e}")
    
    elif args.mode == 'upgrade':
        # Upgrade to object detection
        print("\n" + "="*60)
        print("UPGRADING TO OBJECT DETECTION")
        print("="*60)
        
        if args.video:
            upgrader.upgrade_to_object_detection(args.video)
        elif args.video_dir:
            video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
            for video_file in video_files:
                video_path = os.path.join(args.video_dir, video_file)
                upgrader.upgrade_to_object_detection(video_path)
        else:
            print("❌ Please specify --video or --video-dir for upgrade mode")
    
    elif args.mode == 'recommend':
        # Generate recommendations
        print("\n" + "="*60)
        print("GENERATING IMPROVEMENT RECOMMENDATIONS")
        print("="*60)
        
        performance = upgrader.analyze_current_model_performance()
        recommendations = upgrader.create_training_recommendations(performance)
        
        print(f"\n📄 Recommendations saved to: {upgrader.config['output_directories']['combined_reports']}")
    
    elif args.mode == 'migrate':
        # Create migration plan
        print("\n" + "="*60)
        print("CREATING MIGRATION PLAN")
        print("="*60)
        
        migration_plan = upgrader.generate_migration_plan()
        print(f"\n📋 Migration plan saved to: {upgrader.config['output_directories']['combined_reports']}")
    
    print(f"\n✅ {args.mode.capitalize()} mode completed successfully!")


if __name__ == "__main__":
    main()


# Additional utility functions for your specific use case

class BombusDataPreprocessor:
    """
    Preprocess your existing data for object detection and species classification training
    """
    
    def __init__(self, observations_file: str = "Collection Observations3.xlsx"):
        self.observations_file = observations_file
        self.load_field_observations()
    
    def load_field_observations(self):
        """Load your field observation data"""
        if os.path.exists(self.observations_file):
            self.observations_df = pd.read_excel(self.observations_file)
            print(f"📊 Loaded {len(self.observations_df)} field observations")
        else:
            print(f"⚠️ Field observations file not found: {self.observations_file}")
            self.observations_df = None
    
    def extract_species_annotations(self):
        """
        Extract species-specific annotations from your field notes
        Based on your meeting notes about distinguishing features
        """
        if self.observations_df is None:
            return None
        
        # Look for species mentions in notes
        species_patterns = {
            'bombus_vosnesenskii': ['yellow face', 'yellow.*thorax', 'thin.*yellow.*abdomen'],
            'bombus_californicus': ['no yellow.*face', 'no yellow.*head'],
            'bombus_crotchii': ['no yellow face', 'round', 'yellow.*high.*abdomen']
        }
        
        species_annotations = []
        
        for idx, row in self.observations_df.iterrows():
            notes = str(row.get('Notes', '')).lower()
            activity_notes = str(row.get('Activity on site', '')).lower() + " " + str(row.get('Activity around', '')).lower()
            combined_notes = notes + " " + activity_notes
            
            detected_species = []
            for species, patterns in species_patterns.items():
                for pattern in patterns:
                    if any(keyword in combined_notes for keyword in pattern.split('.*')):
                        detected_species.append(species)
                        break
            
            if detected_species:
                species_annotations.append({
                    'date': row.get('Date'),
                    'week': row.get('Week'),
                    'site': row.get('Site #'),
                    'camera': row.get('Camera #'),
                    'video_id': f"{row.get('Plant_Type', '')}_{row.get('Week', '')}_{row.get('Site #', '')}_{row.get('Camera #', '')}_{row.get('Shift type', '')}",
                    'detected_species': detected_species,
                    'notes': combined_notes,
                    'confidence': 'field_observation'
                })
        
        species_df = pd.DataFrame(species_annotations)
        output_path = "data/species_field_annotations.csv"
        species_df.to_csv(output_path, index=False)
        
        print(f"🔍 Extracted {len(species_annotations)} potential species annotations")
        print(f"💾 Saved to: {output_path}")
        
        return species_df
    
    def prepare_object_detection_labels(self, frame_dir: str = "data/processed/frames/"):
        """
        Prepare object detection labels from your existing binary classification data
        """
        positive_frames_dir = os.path.join(frame_dir, "positive")
        
        if not os.path.exists(positive_frames_dir):
            print(f"⚠️ Positive frames directory not found: {positive_frames_dir}")
            return
        
        # For each positive frame, create a default bounding box annotation
        # This is a placeholder - you'll need to manually annotate actual bounding boxes
        
        frame_files = [f for f in os.listdir(positive_frames_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        yolo_annotations = []
        
        for frame_file in frame_files:
            # Default bounding box (center of image, covering 50% of area)
            # Format: class_id center_x center_y width height (normalized 0-1)
            default_annotation = "1 0.5 0.5 0.5 0.5"  # class 1 = bombus, centered box
            
            annotation_file = frame_file.replace('.jpg', '.txt').replace('.png', '.txt')
            annotation_path = os.path.join(positive_frames_dir, annotation_file)
            
            with open(annotation_path, 'w') as f:
                f.write(default_annotation)
            
            yolo_annotations.append({
                'image_file': frame_file,
                'annotation_file': annotation_file,
                'class': 'bombus',
                'bbox': [0.5, 0.5, 0.5, 0.5],
                'note': 'Default annotation - needs manual refinement'
            })
        
        print(f"📦 Created {len(yolo_annotations)} default YOLO annotations")
        print("⚠️ IMPORTANT: These are placeholder annotations. You need to:")
        print("   1. Use annotation tools like LabelImg or CVAT to create accurate bounding boxes")
        print("   2. Manually verify and adjust all bounding boxes")
        print("   3. Add negative examples (background class)")
        
        # Save annotation summary
        annotation_df = pd.DataFrame(yolo_annotations)
        annotation_df.to_csv("data/object_detection_annotations_summary.csv", index=False)
        
        return yolo_annotations


# Quick setup script
def quick_setup():
    """
    Quick setup for upgrading your existing workflow
    """
    print("🔧 BOMBUS DETECTION WORKFLOW UPGRADE")
    print("="*50)
    
    # Check current repository structure
    required_dirs = ["src", "models", "data", "analysis_output"]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print(f"⚠️ Missing directories: {missing_dirs}")
        print("Please run this script from your bee-monitoring-ml repository root")
        return False
    
    # Check for required files
    required_files = [
        "models/best_bombus_model.pth",
        "Collection Observations3.xlsx",
        "src/correct_video_analyzer.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"⚠️ Missing files: {missing_files}")
    
    # Initialize upgrader
    upgrader = WorkflowUpgrader()
    
    # Run analysis
    print("\n1. Analyzing current performance...")
    performance = upgrader.analyze_current_model_performance()
    
    print("\n2. Generating recommendations...")
    recommendations = upgrader.create_training_recommendations(performance)
    
    print("\n3. Creating migration plan...")
    migration_plan = upgrader.generate_migration_plan()
    
    print("\n4. Preprocessing data for upgrades...")
    preprocessor = BombusDataPreprocessor()
    species_annotations = preprocessor.extract_species_annotations()
    
    print("\n✅ Setup complete! Next steps:")
    print("   1. Review recommendations in combined_analysis_reports/")
    print("   2. Start with Phase 1 of the migration plan")
    print("   3. Consider training object detection model first")
    print("   4. Use synthetic dataset when available for species classification")
    
    return True


# Example usage for your specific workflow
"""
# Basic usage examples:

# 1. Analyze current performance
python workflow_integration.py --mode analyze

# 2. Analyze specific video
python workflow_integration.py --mode analyze --video /path/to/video.mp4

# 3. Upgrade single video to object detection simulation
python workflow_integration.py --mode upgrade --video /path/to/video.mp4

# 4. Generate improvement recommendations
python workflow_integration.py --mode recommend

# 5. Create migration plan
python workflow_integration.py --mode migrate

# 6. Quick setup (run once)
python -c "from workflow_integration import quick_setup; quick_setup()"
"""