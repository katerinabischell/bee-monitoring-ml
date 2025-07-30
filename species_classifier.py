"""
Species Classifier
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class BombusSpeciesClassifier:
    """
    Multi-class classifier for distinguishing between bombus species:
    - Bombus vosnesenskii (Yellow-faced Bumble Bee)
    - Bombus californicus (California Bumble Bee) 
    - Bombus crotchii (Crotch's Bumble Bee)
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Class mapping based on your meeting notes
        self.species_classes = {
            0: 'background',
            1: 'bombus_vosnesenskii',  # Yellow face, yellow on thorax, thin yellow abdomen
            2: 'bombus_californicus',  # No yellow on face/head
            3: 'bombus_crotchii'       # No yellow face, more round, yellow band high on abdomen
        }
        
        self.species_info = {
            'bombus_vosnesenskii': {
                'common_name': 'Yellow-faced Bumble Bee',
                'key_features': ['Yellow face', 'Yellow extends from head to thorax', 'Thin yellow abdomen bands'],
                'notes': 'Sometimes face gets worn down'
            },
            'bombus_californicus': {
                'common_name': 'California Bumble Bee',
                'key_features': ['No yellow on face or top of head'],
                'notes': 'Distinguished by lack of facial yellow coloration'
            },
            'bombus_crotchii': {
                'common_name': "Crotch's Bumble Bee",
                'key_features': ['No yellow face', 'More round body shape', 'Yellow band positioned high on abdomen'],
                'notes': 'Distinctive body proportions and yellow band placement'
            }
        }
        
        # Initialize model
        self.model = self._build_model()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded trained species classifier from {model_path}")
        else:
            print("⚠️ No trained model found. Initialize training first.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _build_model(self) -> nn.Module:
        """
        Build ResNet-50 model for species classification
        """
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify final layer for 4 classes (background + 3 species)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.species_classes))
        
        return model
    
    def predict_species(self, image: np.ndarray) -> Dict:
        """
        Predict bombus species from image
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Transform for model input
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        # Get predictions
        max_prob, predicted_class = torch.max(probabilities, 1)
        confidence = max_prob.item()
        predicted_species = self.species_classes[predicted_class.item()]
        
        # Get all class probabilities
        all_probs = {self.species_classes[i]: probabilities[0][i].item() 
                    for i in range(len(self.species_classes))}
        
        return {
            'predicted_species': predicted_species,
            'confidence': confidence,
            'above_threshold': confidence >= self.confidence_threshold,
            'all_probabilities': all_probs,
            'species_info': self.species_info.get(predicted_species, {})
        }
    
    def analyze_video_for_species(self, video_path: str, sample_interval: int = 30) -> pd.DataFrame:
        """
        Analyze video for species identification
        Sample frames every N seconds for efficiency
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        
        print(f"🎬 Analyzing video for species: {os.path.basename(video_path)}")
        print(f"📊 Duration: {total_duration:.1f}s, FPS: {fps:.1f}")
        print(f"🔍 Sampling every {sample_interval} seconds")
        
        results = []
        frame_interval = int(sample_interval * fps)
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            current_time = frame_idx / fps
            prediction = self.predict_species(frame)
            
            results.append({
                'timestamp': current_time,
                'frame_number': frame_idx,
                'predicted_species': prediction['predicted_species'],
                'confidence': prediction['confidence'],
                'above_threshold': prediction['above_threshold'],
                'prob_vosnesenskii': prediction['all_probabilities']['bombus_vosnesenskii'],
                'prob_californicus': prediction['all_probabilities']['bombus_californicus'],
                'prob_crotchii': prediction['all_probabilities']['bombus_crotchii'],
                'prob_background': prediction['all_probabilities']['background']
            })
        
        cap.release()
        
        df = pd.DataFrame(results)
        
        # Filter for high-confidence bombus detections (exclude background)
        bombus_detections = df[
            (df['above_threshold']) & 
            (df['predicted_species'] != 'background')
        ]
        
        if len(bombus_detections) > 0:
            print(f"\n🐝 Species Detection Summary:")
            species_counts = bombus_detections['predicted_species'].value_counts()
            for species, count in species_counts.items():
                print(f"   {species}: {count} detections")
                
            # Dominant species
            dominant_species = species_counts.index[0] if len(species_counts) > 0 else "None"
            print(f"\n👑 Dominant species: {dominant_species}")
        else:
            print("⚠️ No high-confidence bombus detections found")
        
        return df
    
    def create_species_timeline_visualization(self, results_df: pd.DataFrame, output_path: str):
        """
        Create timeline visualization of species detections
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Bombus Species Detection Timeline', fontsize=16)
        
        # Filter for bombus detections only
        bombus_data = results_df[results_df['predicted_species'] != 'background'].copy()
        
        if len(bombus_data) == 0:
            print("No bombus detections to visualize")
            return
        
        # 1. Species detection timeline
        species_colors = {
            'bombus_vosnesenskii': 'gold',
            'bombus_californicus': 'orange', 
            'bombus_crotchii': 'red'
        }
        
        for species in species_colors.keys():
            species_data = bombus_data[bombus_data['predicted_species'] == species]
            if len(species_data) > 0:
                axes[0].scatter(species_data['timestamp'], [species]*len(species_data), 
                              c=species_colors[species], s=60, alpha=0.7, 
                              label=species.replace('bombus_', 'B. '))
        
        axes[0].set_ylabel('Species')
        axes[0].set_title('Species Detections Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Confidence scores over time
        axes[1].scatter(bombus_data['timestamp'], bombus_data['confidence'], 
                       c=[species_colors.get(species, 'gray') for species in bombus_data['predicted_species']], 
                       alpha=0.7, s=40)
        axes[1].axhline(y=self.confidence_threshold, color='red', linestyle='--', 
                       label=f'Confidence Threshold ({self.confidence_threshold})')
        axes[1].set_ylabel('Confidence Score')
        axes[1].set_title('Detection Confidence Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Species probability heatmap over time
        prob_data = bombus_data[['timestamp', 'prob_vosnesenskii', 'prob_californicus', 'prob_crotchii']].copy()
        prob_data.set_index('timestamp', inplace=True)
        
        # Resample to create regular time intervals for heatmap
        if len(prob_data) > 1:
            prob_data_resampled = prob_data.resample('30S').mean().fillna(0)
            
            im = axes[2].imshow(prob_data_resampled.T, aspect='auto', cmap='YlOrRd', 
                              extent=[prob_data_resampled.index.min(), prob_data_resampled.index.max(), 0, 3])
            axes[2].set_yticks([0.5, 1.5, 2.5])
            axes[2].set_yticklabels(['B. vosnesenskii', 'B. californicus', 'B. crotchii'])
            axes[2].set_xlabel('Time (seconds)')
            axes[2].set_title('Species Probability Heatmap')
            plt.colorbar(im, ax=axes[2], label='Probability')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_species_report(self, results_df: pd.DataFrame, video_name: str) -> Dict:
        """
        Generate comprehensive species identification report
        """
        # Filter high-confidence bombus detections
        bombus_detections = results_df[
            (results_df['above_threshold']) & 
            (results_df['predicted_species'] != 'background')
        ]
        
        if len(bombus_detections) == 0:
            return {
                'video_name': video_name,
                'total_detections': 0,
                'species_identified': [],
                'dominant_species': None,
                'detection_summary': "No high-confidence bombus species detected"
            }
        
        # Species counts and percentages
        species_counts = bombus_detections['predicted_species'].value_counts()
        total_bombus_detections = len(bombus_detections)
        
        species_summary = {}
        for species, count in species_counts.items():
            percentage = (count / total_bombus_detections) * 100
            avg_confidence = bombus_detections[bombus_detections['predicted_species'] == species]['confidence'].mean()
            
            species_summary[species] = {
                'count': count,
                'percentage': percentage,
                'avg_confidence': avg_confidence,
                'common_name': self.species_info.get(species, {}).get('common_name', species),
                'key_features': self.species_info.get(species, {}).get('key_features', [])
            }
        
        # Dominant species (most frequent)
        dominant_species = species_counts.index[0] if len(species_counts) > 0 else None
        
        # Temporal patterns
        first_detection = bombus_detections['timestamp'].min()
        last_detection = bombus_detections['timestamp'].max()
        detection_span = last_detection - first_detection
        
        report = {
            'video_name': video_name,
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_detections': total_bombus_detections,
            'species_identified': list(species_counts.index),
            'dominant_species': dominant_species,
            'species_summary': species_summary,
            'temporal_patterns': {
                'first_detection_time': first_detection,
                'last_detection_time': last_detection,
                'detection_span_seconds': detection_span,
                'detection_frequency': total_bombus_detections / (detection_span / 60) if detection_span > 0 else 0  # per minute
            },
            'confidence_stats': {
                'mean_confidence': bombus_detections['confidence'].mean(),
                'min_confidence': bombus_detections['confidence'].min(),
                'max_confidence': bombus_detections['confidence'].max()
            }
        }
        
        return report


class VideoSpeciesBatchProcessor:
    """
    Process multiple videos for species identification
    """
    
    def __init__(self, classifier: BombusSpeciesClassifier):
        self.classifier = classifier
    
    def process_video_directory(self, video_dir: str, output_dir: str, sample_interval: int = 30):
        """
        Process all videos for species identification
        """
        os.makedirs(output_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        all_reports = []
        
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            print(f"\n{'='*60}")
            print(f"Processing: {video_file}")
            
            # Analyze video for species
            results_df = self.classifier.analyze_video_for_species(video_path, sample_interval)
            
            # Save detailed results
            results_path = os.path.join(output_dir, f"{video_file.replace('.mp4', '')}_species_detections.csv")
            results_df.to_csv(results_path, index=False)
            
            # Generate species report
            report = self.classifier.generate_species_report(results_df, video_file)
            
            # Save individual report
            report_path = os.path.join(output_dir, f"{video_file.replace('.mp4', '')}_species_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Create visualization
            viz_path = os.path.join(output_dir, f"{video_file.replace('.mp4', '')}_species_timeline.png")
            self.classifier.create_species_timeline_visualization(results_df, viz_path)
            
            all_reports.append(report)
            
            # Print summary
            if report['total_detections'] > 0:
                print(f"✅ {report['total_detections']} bombus detections")
                print(f"🏆 Dominant species: {report['dominant_species']}")
                for species, data in report['species_summary'].items():
                    print(f"   - {data['common_name']}: {data['count']} ({data['percentage']:.1f}%)")
            else:
                print("❌ No bombus species detected")
        
        # Generate combined analysis
        self.generate_combined_species_analysis(all_reports, output_dir)
        
        return all_reports
    
    def generate_combined_species_analysis(self, all_reports: List[Dict], output_dir: str):
        """
        Generate combined analysis across all videos
        """
        # Aggregate data
        total_videos = len(all_reports)
        videos_with_detections = sum(1 for r in all_reports if r['total_detections'] > 0)
        
        # Species occurrence across videos
        all_species = set()
        species_video_counts = Counter()
        species_total_detections = Counter()
        
        for report in all_reports:
            for species in report['species_identified']:
                all_species.add(species)
                species_video_counts[species] += 1
                species_total_detections[species] += report['species_summary'][species]['count']
        
        # Dominant species per video
        dominant_species_counts = Counter([r['dominant_species'] for r in all_reports if r['dominant_species']])
        
        combined_analysis = {
            'analysis_summary': {
                'total_videos_processed': total_videos,
                'videos_with_bombus_detections': videos_with_detections,
                'detection_rate': videos_with_detections / total_videos if total_videos > 0 else 0,
                'total_bombus_detections': sum(r['total_detections'] for r in all_reports),
                'species_found': list(all_species)
            },
            'species_statistics': {
                'species_occurrence_by_video': dict(species_video_counts),
                'total_detections_by_species': dict(species_total_detections),
                'dominant_species_frequency': dict(dominant_species_counts)
            },
            'per_video_results': all_reports
        }
        
        # Calculate percentages
        for species in all_species:
            video_occurrence_pct = (species_video_counts[species] / total_videos) * 100
            combined_analysis['species_statistics'][f'{species}_video_occurrence_percentage'] = video_occurrence_pct
        
        # Save combined analysis
        combined_path = os.path.join(output_dir, 'combined_species_analysis.json')
        with open(combined_path, 'w') as f:
            json.dump(combined_analysis, f, indent=2, default=str)
        
        # Create summary visualization
        self.create_summary_visualization(combined_analysis, output_dir)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 COMBINED SPECIES ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total videos processed: {total_videos}")
        print(f"Videos with bombus: {videos_with_detections} ({videos_with_detections/total_videos*100:.1f}%)")
        print(f"Total bombus detections: {combined_analysis['analysis_summary']['total_bombus_detections']}")
        
        if len(all_species) > 0:
            print(f"\nSpecies detected:")
            for species in sorted(all_species):
                common_name = self.classifier.species_info.get(species, {}).get('common_name', species)
                video_count = species_video_counts[species]
                total_detections = species_total_detections[species]
                video_pct = (video_count / total_videos) * 100
                print(f"  🐝 {common_name}:")
                print(f"     - Found in {video_count}/{total_videos} videos ({video_pct:.1f}%)")
                print(f"     - Total detections: {total_detections}")
        
        if len(dominant_species_counts) > 0:
            print(f"\nMost common dominant species:")
            for species, count in dominant_species_counts.most_common(3):
                common_name = self.classifier.species_info.get(species, {}).get('common_name', species)
                print(f"  👑 {common_name}: dominant in {count} videos")
    
    def create_summary_visualization(self, combined_analysis: Dict, output_dir: str):
        """
        Create summary visualization across all videos
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bombus Species Analysis - All Videos Summary', fontsize=16)
        
        species_stats = combined_analysis['species_statistics']
        
        # 1. Species occurrence by video
        if 'species_occurrence_by_video' in species_stats:
            species_names = list(species_stats['species_occurrence_by_video'].keys())
            occurrence_counts = list(species_stats['species_occurrence_by_video'].values())
            
            if len(species_names) > 0:
                colors = ['gold', 'orange', 'red'][:len(species_names)]
                bars = axes[0,0].bar(range(len(species_names)), occurrence_counts, color=colors, alpha=0.7)
                axes[0,0].set_title('Species Occurrence Across Videos')
                axes[0,0].set_xlabel('Species')
                axes[0,0].set_ylabel('Number of Videos')
                axes[0,0].set_xticks(range(len(species_names)))
                axes[0,0].set_xticklabels([name.replace('bombus_', 'B. ') for name in species_names], 
                                         rotation=45, ha='right')
                
                # Add count labels on bars
                for bar, count in zip(bars, occurrence_counts):
                    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                  str(count), ha='center', va='bottom')
        
        # 2. Total detections by species
        if 'total_detections_by_species' in species_stats:
            species_names = list(species_stats['total_detections_by_species'].keys())
            detection_counts = list(species_stats['total_detections_by_species'].values())
            
            if len(species_names) > 0:
                colors = ['gold', 'orange', 'red'][:len(species_names)]
                wedges, texts, autotexts = axes[0,1].pie(detection_counts, labels=[name.replace('bombus_', 'B. ') for name in species_names], 
                                                        colors=colors, autopct='%1.1f%%', startangle=90)
                axes[0,1].set_title('Total Detections by Species')
        
        # 3. Dominant species frequency
        if 'dominant_species_frequency' in species_stats:
            dominant_species = list(species_stats['dominant_species_frequency'].keys())
            dominant_counts = list(species_stats['dominant_species_frequency'].values())
            
            if len(dominant_species) > 0:
                colors = ['gold', 'orange', 'red'][:len(dominant_species)]
                bars = axes[1,0].bar(range(len(dominant_species)), dominant_counts, color=colors, alpha=0.7)
                axes[1,0].set_title('Dominant Species Frequency')
                axes[1,0].set_xlabel('Species')
                axes[1,0].set_ylabel('Videos Where Dominant')
                axes[1,0].set_xticks(range(len(dominant_species)))
                axes[1,0].set_xticklabels([name.replace('bombus_', 'B. ') for name in dominant_species], 
                                         rotation=45, ha='right')
                
                # Add count labels
                for bar, count in zip(bars, dominant_counts):
                    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                  str(count), ha='center', va='bottom')
        
        # 4. Detection rate summary
        total_videos = combined_analysis['analysis_summary']['total_videos_processed']
        videos_with_bombus = combined_analysis['analysis_summary']['videos_with_bombus_detections']
        videos_without_bombus = total_videos - videos_with_bombus
        
        detection_summary = [videos_with_bombus, videos_without_bombus]
        labels = ['Videos with Bombus', 'Videos without Bombus']
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = axes[1,1].pie(detection_summary, labels=labels, colors=colors, 
                                                autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Overall Detection Success Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'species_analysis_summary.png'), dpi=300, bbox_inches='tight')
        plt.show()


# Training pipeline for species classifier
class SpeciesClassifierTrainer:
    """
    Training pipeline for bombus species classification
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_synthetic_dataset(self, synthetic_data_dir: str):
        """
        Prepare training dataset including synthetic data
        Expected directory structure:
        synthetic_data_dir/
        ├── bombus_vosnesenskii/
        ├── bombus_californicus/
        ├── bombus_crotchii/
        └── background/
        """
        print("🔧 Preparing synthetic dataset for species classification...")
        
        # Data augmentation for synthetic data
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets (you'll need to implement ImageFolder or custom dataset)
        # This is a template - you'll need to adapt based on your synthetic data format
        
        print("✅ Synthetic dataset prepared")
        return train_transform, val_transform


# Usage Example
if __name__ == "__main__":
    # Initialize species classifier
    classifier = BombusSpeciesClassifier(
        model_path='models/bombus_species_classifier.pth',
        confidence_threshold=0.7
    )
    
    # Analyze single video for species
    video_path = "/path/to/your/video.mp4"
    results = classifier.analyze_video_for_species(video_path, sample_interval=30)
    
    # Generate species report
    report = classifier.generate_species_report(results, "test_video.mp4")
    print(json.dumps(report, indent=2, default=str))
    
    # Create visualization
    classifier.create_species_timeline_visualization(results, 'species_timeline.png')
    
    # Process entire directory
    processor = VideoSpeciesBatchProcessor(classifier)
    all_reports = processor.process_video_directory(
        video_dir="/path/to/video/directory",
        output_dir="species_analysis_results",
        sample_interval=30
    )
    
    print("\n✅ Species analysis complete! Check species_analysis_results/ for detailed outputs.")