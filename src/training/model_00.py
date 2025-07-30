#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pollinator Dataset Analysis and Preparation Tool
Author: Katerina Bischel
Project: Endangered Coastal Plant Pollinator Monitoring

This script analyzes ollection observations and prepares recommendations
for machine learning model development.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from collections import Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PollinatorDatasetAnalyzer:
    """Comprehensive analysis tool for pollinator observation data"""
    
    def __init__(self, observations_file):
        """Initialize with Excel file containing observations"""
        self.observations_file = observations_file
        self.birds_beak_df = None
        self.milkvetch_df = None
        self.analysis_results = {}
        
        self.load_data()
    
    def load_data(self):
        """Load observation data from Excel file"""
        try:
            # Load both sheets
            self.birds_beak_df = pd.read_excel(self.observations_file, sheet_name='Birds Beak')
            self.milkvetch_df = pd.read_excel(self.observations_file, sheet_name='Ventura Milkvetch')
            
            logger.info(f"Loaded Birds Beak observations: {len(self.birds_beak_df)} records")
            logger.info(f"Loaded Ventura Milkvetch observations: {len(self.milkvetch_df)} records")
            
            # Clean and standardize data
            self._clean_data()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _clean_data(self):
        """Clean and standardize the observation data"""
        for df_name, df in [("Birds Beak", self.birds_beak_df), ("Ventura Milkvetch", self.milkvetch_df)]:
            logger.info(f"Cleaning {df_name} data...")
            
            # Convert activity columns to string and handle NaN
            df['Activity on site'] = df['Activity on site'].fillna('none').astype(str).str.lower()
            df['Activity around '] = df['Activity around '].fillna('none').astype(str).str.lower()
            df['Notes'] = df['Notes'].fillna('').astype(str)
            
            # Standardize 'none' variations
            df['Activity on site'] = df['Activity on site'].replace(['nan', 'na', ''], 'none')
            df['Activity around '] = df['Activity around '].replace(['nan', 'na', ''], 'none')
    
    def analyze_bombus_observations(self):
        """Analyze bombus-related observations in detail"""
        logger.info("Analyzing bombus observations...")
        
        analysis = {}
        
        for plant_type, df in [("Birds Beak", self.birds_beak_df), ("Ventura Milkvetch", self.milkvetch_df)]:
            plant_analysis = {}
            
            # Define bombus-related keywords
            bombus_keywords = [
                'bombus', 'b. californicus', 'b. vosnesenskii', 'bombus v', 'bombus vos',
                'bumble bee', 'bumblebee'
            ]
            
            # Find bombus observations
            bombus_on_site = df['Activity on site'].str.contains('|'.join(bombus_keywords), na=False)
            bombus_around = df['Activity around '].str.contains('|'.join(bombus_keywords), na=False)
            bombus_any = bombus_on_site | bombus_around
            
            plant_analysis['total_observations'] = len(df)
            plant_analysis['bombus_on_site'] = bombus_on_site.sum()
            plant_analysis['bombus_around_site'] = bombus_around.sum()
            plant_analysis['bombus_any_activity'] = bombus_any.sum()
            plant_analysis['bombus_percentage'] = (bombus_any.sum() / len(df)) * 100
            
            # Extract specific bombus species mentions
            vosnesenskii_mentions = df['Activity on site'].str.contains('vosnesenskii|bombus v|bombus vos', na=False) | \
                                   df['Activity around '].str.contains('vosnesenskii|bombus v|bombus vos', na=False)
            californicus_mentions = df['Activity on site'].str.contains('californicus', na=False) | \
                                   df['Activity around '].str.contains('californicus', na=False)
            
            plant_analysis['vosnesenskii_observations'] = vosnesenskii_mentions.sum()
            plant_analysis['californicus_observations'] = californicus_mentions.sum()
            
            # Analyze by time periods
            plant_analysis['bombus_by_shift'] = df[bombus_any]['Shift type'].value_counts().to_dict()
            plant_analysis['bombus_by_week'] = df[bombus_any]['Week '].value_counts().to_dict()
            plant_analysis['bombus_by_site'] = df[bombus_any]['Site #'].value_counts().to_dict()
            
            # Weather conditions during bombus observations
            if 'Weather' in df.columns:
                plant_analysis['bombus_weather_conditions'] = df[bombus_any]['Weather'].value_counts().to_dict()
            
            # Video duration analysis for bombus observations
            plant_analysis['bombus_video_durations'] = df[bombus_any]['Video Duration'].value_counts().to_dict()
            
            analysis[plant_type] = plant_analysis
        
        self.analysis_results['bombus_analysis'] = analysis
        return analysis
    
    def analyze_all_pollinator_activity(self):
        """Analyze all types of pollinator activity"""
        logger.info("Analyzing all pollinator activity...")
        
        analysis = {}
        
        for plant_type, df in [("Birds Beak", self.birds_beak_df), ("Ventura Milkvetch", self.milkvetch_df)]:
            plant_analysis = {}
            
            # Define broader pollinator keywords
            pollinator_keywords = [
                'bombus', 'bee', 'honeybee', 'wasp', 'butterfly', 'moth', 'fly', 'beetle',
                'pollinator', 'insect', 'hover'
            ]
            
            # Find any pollinator activity
            activity_on = df['Activity on site'] != 'none'
            activity_around = df['Activity around '] != 'none'
            any_activity = activity_on | activity_around
            
            # Find specific pollinator activity
            pollinator_on = df['Activity on site'].str.contains('|'.join(pollinator_keywords), na=False)
            pollinator_around = df['Activity around '].str.contains('|'.join(pollinator_keywords), na=False)
            pollinator_any = pollinator_on | pollinator_around
            
            plant_analysis['total_observations'] = len(df)
            plant_analysis['any_activity'] = any_activity.sum()
            plant_analysis['pollinator_activity'] = pollinator_any.sum()
            plant_analysis['activity_percentage'] = (any_activity.sum() / len(df)) * 100
            plant_analysis['pollinator_percentage'] = (pollinator_any.sum() / len(df)) * 100
            
            # Extract all unique activities for classification
            all_activities = []
            all_activities.extend(df[activity_on]['Activity on site'].unique())
            all_activities.extend(df[activity_around]['Activity around '].unique())
            
            # Remove 'none' and clean up
            unique_activities = [act for act in set(all_activities) if act != 'none' and pd.notna(act)]
            plant_analysis['unique_activities'] = sorted(unique_activities)
            
            # Count activity types
            activity_counter = Counter()
            for _, row in df.iterrows():
                if row['Activity on site'] != 'none':
                    activity_counter[row['Activity on site']] += 1
                if row['Activity around '] != 'none':
                    activity_counter[row['Activity around ']] += 1
            
            plant_analysis['activity_counts'] = dict(activity_counter.most_common(20))
            
            analysis[plant_type] = plant_analysis
        
        self.analysis_results['pollinator_analysis'] = analysis
        return analysis
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in observations"""
        logger.info("Analyzing temporal patterns...")
        
        analysis = {}
        
        for plant_type, df in [("Birds Beak", self.birds_beak_df), ("Ventura Milkvetch", self.milkvetch_df)]:
            plant_analysis = {}
            
            # Week analysis
            week_counts = df['Week '].value_counts().sort_index()
            plant_analysis['observations_by_week'] = week_counts.to_dict()
            
            # Shift type analysis
            shift_counts = df['Shift type'].value_counts()
            plant_analysis['observations_by_shift'] = shift_counts.to_dict()
            
            # Site analysis
            site_counts = df['Site #'].value_counts().sort_index()
            plant_analysis['observations_by_site'] = site_counts.to_dict()
            
            # Camera analysis
            camera_counts = df['Camera #'].value_counts().sort_index()
            plant_analysis['observations_by_camera'] = camera_counts.to_dict()
            
            # Video duration patterns
            duration_counts = df['Video Duration'].value_counts()
            plant_analysis['video_duration_distribution'] = duration_counts.to_dict()
            
            analysis[plant_type] = plant_analysis
        
        self.analysis_results['temporal_analysis'] = analysis
        return analysis
    
    def create_training_recommendations(self):
        """Generate recommendations for ML model training"""
        logger.info("Generating training recommendations...")
        
        recommendations = {}
        
        # Overall dataset characteristics
        total_birds_beak = len(self.birds_beak_df)
        total_milkvetch = len(self.milkvetch_df)
        
        bombus_birds_beak = self.analysis_results['bombus_analysis']['Birds Beak']['bombus_any_activity']
        bombus_milkvetch = self.analysis_results['bombus_analysis']['Ventura Milkvetch']['bombus_any_activity']
        
        recommendations['dataset_overview'] = {
            'total_birds_beak_videos': total_birds_beak,
            'total_milkvetch_videos': total_milkvetch,
            'total_videos': total_birds_beak + total_milkvetch,
            'bombus_positive_birds_beak': bombus_birds_beak,
            'bombus_positive_milkvetch': bombus_milkvetch,
            'total_bombus_positive': bombus_birds_beak + bombus_milkvetch
        }
        
        # Class balance analysis
        total_positive = bombus_birds_beak + bombus_milkvetch
        total_videos = total_birds_beak + total_milkvetch
        positive_ratio = total_positive / total_videos
        
        recommendations['class_balance'] = {
            'positive_samples': total_positive,
            'negative_samples': total_videos - total_positive,
            'positive_ratio': positive_ratio,
            'imbalance_severity': 'severe' if positive_ratio < 0.1 else 'moderate' if positive_ratio < 0.3 else 'balanced'
        }
        
        # Data augmentation recommendations
        if positive_ratio < 0.2:
            recommendations['augmentation_strategy'] = {
                'priority': 'high',
                'techniques': [
                    'horizontal_flip',
                    'rotation',
                    'brightness_adjustment',
                    'contrast_adjustment',
                    'random_crop',
                    'gaussian_noise'
                ],
                'target_balance_ratio': 0.4
            }
        else:
            recommendations['augmentation_strategy'] = {
                'priority': 'medium',
                'techniques': [
                    'horizontal_flip',
                    'rotation',
                    'brightness_adjustment'
                ]
            }
        
        # Model architecture recommendations
        if total_videos > 1000:
            recommendations['model_architecture'] = {
                'complexity': 'medium_to_high',
                'suggestions': [
                    'ResNet-18 or ResNet-34 transfer learning',
                    'Custom CNN with 4-6 convolutional layers',
                    'EfficientNet-B0 transfer learning'
                ]
            }
        else:
            recommendations['model_architecture'] = {
                'complexity': 'simple_to_medium',
                'suggestions': [
                    'Simple CNN with 3-4 convolutional layers',
                    'ResNet-18 transfer learning',
                    'MobileNet transfer learning'
                ]
            }
        
        # Training strategy
        recommendations['training_strategy'] = {
            'batch_size': min(32, max(8, total_positive // 10)),
            'learning_rate': 0.001,
            'epochs': 50 if total_positive > 50 else 30,
            'validation_split': 0.2,
            'early_stopping': True,
            'class_weights': True if positive_ratio < 0.3 else False
        }
        
        # Frame extraction strategy
        avg_duration = 25  # Approximate average duration in minutes
        frames_per_minute = 2  # Conservative frame extraction rate
        
        recommendations['frame_extraction'] = {
            'frames_per_video': frames_per_minute * avg_duration,
            'extraction_method': 'uniform',
            'target_frame_count': total_videos * frames_per_minute * avg_duration,
            'positive_frame_estimate': total_positive * frames_per_minute * avg_duration
        }
        
        self.analysis_results['training_recommendations'] = recommendations
        return recommendations
    
    def create_visualizations(self, output_dir='analysis_output'):
        """Create comprehensive visualizations of the dataset"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating visualizations in {output_dir}")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Bombus observations comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bombus activity by plant type
        plant_types = ['Birds Beak', 'Ventura Milkvetch']
        bombus_counts = [
            self.analysis_results['bombus_analysis']['Birds Beak']['bombus_any_activity'],
            self.analysis_results['bombus_analysis']['Ventura Milkvetch']['bombus_any_activity']
        ]
        total_counts = [
            self.analysis_results['bombus_analysis']['Birds Beak']['total_observations'],
            self.analysis_results['bombus_analysis']['Ventura Milkvetch']['total_observations']
        ]
        
        axes[0, 0].bar(plant_types, bombus_counts, alpha=0.7, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Bombus Observations by Plant Type')
        axes[0, 0].set_ylabel('Number of Videos with Bombus')
        
        # Bombus percentage by plant type
        bombus_percentages = [
            self.analysis_results['bombus_analysis']['Birds Beak']['bombus_percentage'],
            self.analysis_results['bombus_analysis']['Ventura Milkvetch']['bombus_percentage']
        ]
        
        axes[0, 1].bar(plant_types, bombus_percentages, alpha=0.7, color=['lightgreen', 'gold'])
        axes[0, 1].set_title('Bombus Detection Rate by Plant Type')
        axes[0, 1].set_ylabel('Percentage of Videos with Bombus')
        
        # Temporal patterns - shift type for Birds Beak
        bb_shift_data = self.analysis_results['bombus_analysis']['Birds Beak']['bombus_by_shift']
        if bb_shift_data:
            shifts = list(bb_shift_data.keys())
            counts = list(bb_shift_data.values())
            axes[1, 0].bar(shifts, counts, alpha=0.7, color='mediumpurple')
            axes[1, 0].set_title('Bombus Activity by Time of Day (Birds Beak)')
            axes[1, 0].set_ylabel('Number of Observations')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Site distribution for bombus observations
        bb_site_data = self.analysis_results['bombus_analysis']['Birds Beak']['bombus_by_site']
        if bb_site_data:
            sites = list(bb_site_data.keys())
            counts = list(bb_site_data.values())
            axes[1, 1].bar([f"Site {s}" for s in sites], counts, alpha=0.7, color='orange')
            axes[1, 1].set_title('Bombus Activity by Site (Birds Beak)')
            axes[1, 1].set_ylabel('Number of Observations')
        
        plt.tight_layout()
        plt.savefig(output_dir / "bombus_analysis_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temporal distribution analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Weekly distribution for Birds Beak
        bb_weekly = self.analysis_results['temporal_analysis']['Birds Beak']['observations_by_week']
        weeks = list(bb_weekly.keys())
        counts = list(bb_weekly.values())
        
        axes[0, 0].plot(weeks, counts, marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Birds Beak Observations by Week')
        axes[0, 0].set_xlabel('Week')
        axes[0, 0].set_ylabel('Number of Videos')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weekly distribution for Milkvetch
        mv_weekly = self.analysis_results['temporal_analysis']['Ventura Milkvetch']['observations_by_week']
        weeks_mv = list(mv_weekly.keys())
        counts_mv = list(mv_weekly.values())
        
        axes[0, 1].plot(weeks_mv, counts_mv, marker='s', linewidth=2, markersize=6, color='red')
        axes[0, 1].set_title('Ventura Milkvetch Observations by Week')
        axes[0, 1].set_xlabel('Week')
        axes[0, 1].set_ylabel('Number of Videos')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Shift distribution comparison
        bb_shifts = self.analysis_results['temporal_analysis']['Birds Beak']['observations_by_shift']
        mv_shifts = self.analysis_results['temporal_analysis']['Ventura Milkvetch']['observations_by_shift']
        
        shift_types = list(set(list(bb_shifts.keys()) + list(mv_shifts.keys())))
        bb_shift_counts = [bb_shifts.get(shift, 0) for shift in shift_types]
        mv_shift_counts = [mv_shifts.get(shift, 0) for shift in shift_types]
        
        x = np.arange(len(shift_types))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, bb_shift_counts, width, label='Birds Beak', alpha=0.7)
        axes[1, 0].bar(x + width/2, mv_shift_counts, width, label='Ventura Milkvetch', alpha=0.7)
        axes[1, 0].set_title('Observations by Shift Type')
        axes[1, 0].set_xlabel('Shift Type')
        axes[1, 0].set_ylabel('Number of Videos')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(shift_types)
        axes[1, 0].legend()
        
        # Video duration distribution
        bb_durations = self.analysis_results['temporal_analysis']['Birds Beak']['video_duration_distribution']
        duration_labels = list(bb_durations.keys())
        duration_counts = list(bb_durations.values())
        
        axes[1, 1].pie(duration_counts, labels=duration_labels, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Video Duration Distribution (Birds Beak)')
        
        plt.tight_layout()
        plt.savefig(output_dir / "temporal_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Dataset balance and recommendations visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class balance pie chart
        recommendations = self.analysis_results['training_recommendations']
        positive_samples = recommendations['dataset_overview']['total_bombus_positive']
        negative_samples = recommendations['class_balance']['negative_samples']
        
        axes[0, 0].pie([positive_samples, negative_samples], 
                       labels=['Bombus Present', 'No Bombus'], 
                       autopct='%1.1f%%', 
                       colors=['lightgreen', 'lightcoral'],
                       startangle=90)
        axes[0, 0].set_title('Dataset Class Balance')
        
        # Sample distribution by plant type
        bb_total = recommendations['dataset_overview']['total_birds_beak_videos']
        mv_total = recommendations['dataset_overview']['total_milkvetch_videos']
        bb_positive = recommendations['dataset_overview']['bombus_positive_birds_beak']
        mv_positive = recommendations['dataset_overview']['bombus_positive_milkvetch']
        
        plant_labels = ['Birds Beak', 'Ventura Milkvetch']
        positive_counts = [bb_positive, mv_positive]
        negative_counts = [bb_total - bb_positive, mv_total - mv_positive]
        
        x = np.arange(len(plant_labels))
        width = 0.35
        
        axes[0, 1].bar(x, negative_counts, width, label='No Bombus', color='lightcoral', alpha=0.7)
        axes[0, 1].bar(x, positive_counts, width, bottom=negative_counts, label='Bombus Present', 
                       color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Sample Distribution by Plant Type')
        axes[0, 1].set_xlabel('Plant Type')
        axes[0, 1].set_ylabel('Number of Videos')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(plant_labels)
        axes[0, 1].legend()
        
        # Training recommendations summary
        rec_text = f"""
Training Recommendations Summary:

Total Videos: {recommendations['dataset_overview']['total_videos']}
Positive Samples: {positive_samples} ({recommendations['class_balance']['positive_ratio']:.1%})
Negative Samples: {negative_samples}

Imbalance Severity: {recommendations['class_balance']['imbalance_severity'].title()}
Augmentation Priority: {recommendations['augmentation_strategy']['priority'].title()}

Recommended:
• Batch Size: {recommendations['training_strategy']['batch_size']}
• Learning Rate: {recommendations['training_strategy']['learning_rate']}
• Epochs: {recommendations['training_strategy']['epochs']}
• Class Weights: {recommendations['training_strategy']['class_weights']}

Frame Extraction:
• Frames per Video: {recommendations['frame_extraction']['frames_per_video']}
• Estimated Total Frames: {recommendations['frame_extraction']['target_frame_count']:,}
• Estimated Positive Frames: {recommendations['frame_extraction']['positive_frame_estimate']:,}
        """
        
        axes[1, 0].text(0.05, 0.95, rec_text, transform=axes[1, 0].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Training Recommendations')
        
        # Species-specific observations
        bb_vos = self.analysis_results['bombus_analysis']['Birds Beak']['vosnesenskii_observations']
        bb_cal = self.analysis_results['bombus_analysis']['Birds Beak']['californicus_observations']
        mv_vos = self.analysis_results['bombus_analysis']['Ventura Milkvetch']['vosnesenskii_observations']
        mv_cal = self.analysis_results['bombus_analysis']['Ventura Milkvetch']['californicus_observations']
        
        species_data = {
            'B. vosnesenskii': [bb_vos, mv_vos],
            'B. californicus': [bb_cal, mv_cal]
        }
        
        x = np.arange(len(plant_labels))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, species_data['B. vosnesenskii'], width, 
                       label='B. vosnesenskii', alpha=0.7, color='steelblue')
        axes[1, 1].bar(x + width/2, species_data['B. californicus'], width, 
                       label='B. californicus', alpha=0.7, color='darkorange')
        axes[1, 1].set_title('Bombus Species Observations by Plant Type')
        axes[1, 1].set_xlabel('Plant Type')
        axes[1, 1].set_ylabel('Number of Observations')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(plant_labels)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_recommendations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations saved successfully!")
    
    def generate_comprehensive_report(self, output_dir='analysis_output'):
        """Generate comprehensive analysis report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating comprehensive report...")
        
        # Run all analyses
        self.analyze_bombus_observations()
        self.analyze_all_pollinator_activity()
        self.analyze_temporal_patterns()
        self.create_training_recommendations()
        
        # Create visualizations
        self.create_visualizations(output_dir)
        
        # Save detailed results
        with open(output_dir / "detailed_analysis_results.json", 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        # Create summary report
        self._create_summary_report(output_dir)
        
        logger.info(f"Comprehensive report generated in {output_dir}")
    
    def _create_summary_report(self, output_dir):
        """Create markdown summary report"""
        report_content = f"""# Pollinator Dataset Analysis Report

## Dataset Overview

### Birds Beak (Chloropyron maritimum)
- **Total Videos**: {self.analysis_results['bombus_analysis']['Birds Beak']['total_observations']}
- **Videos with Bombus**: {self.analysis_results['bombus_analysis']['Birds Beak']['bombus_any_activity']}
- **Detection Rate**: {self.analysis_results['bombus_analysis']['Birds Beak']['bombus_percentage']:.1f}%

### Ventura Milkvetch (Astragalus pycnostachyus)
- **Total Videos**: {self.analysis_results['bombus_analysis']['Ventura Milkvetch']['total_observations']}
- **Videos with Bombus**: {self.analysis_results['bombus_analysis']['Ventura Milkvetch']['bombus_any_activity']}
- **Detection Rate**: {self.analysis_results['bombus_analysis']['Ventura Milkvetch']['bombus_percentage']:.1f}%

## Species-Specific Observations

### B. vosnesenskii (Target Species)
- **Birds Beak**: {self.analysis_results['bombus_analysis']['Birds Beak']['vosnesenskii_observations']} observations
- **Ventura Milkvetch**: {self.analysis_results['bombus_analysis']['Ventura Milkvetch']['vosnesenskii_observations']} observations

### B. californicus
- **Birds Beak**: {self.analysis_results['bombus_analysis']['Birds Beak']['californicus_observations']} observations
- **Ventura Milkvetch**: {self.analysis_results['bombus_analysis']['Ventura Milkvetch']['californicus_observations']} observations

## Machine Learning Recommendations

### Dataset Characteristics
- **Total Videos**: {self.analysis_results['training_recommendations']['dataset_overview']['total_videos']}
- **Positive Samples**: {self.analysis_results['training_recommendations']['dataset_overview']['total_bombus_positive']}
- **Class Balance**: {self.analysis_results['training_recommendations']['class_balance']['positive_ratio']:.1%} positive
- **Imbalance Severity**: {self.analysis_results['training_recommendations']['class_balance']['imbalance_severity'].title()}

### Training Strategy
- **Recommended Batch Size**: {self.analysis_results['training_recommendations']['training_strategy']['batch_size']}
- **Learning Rate**: {self.analysis_results['training_recommendations']['training_strategy']['learning_rate']}
- **Epochs**: {self.analysis_results['training_recommendations']['training_strategy']['epochs']}
- **Use Class Weights**: {self.analysis_results['training_recommendations']['training_strategy']['class_weights']}

### Data Augmentation
- **Priority**: {self.analysis_results['training_recommendations']['augmentation_strategy']['priority'].title()}
- **Recommended Techniques**: {', '.join(self.analysis_results['training_recommendations']['augmentation_strategy']['techniques'])}

### Frame Extraction Strategy
- **Frames per Video**: {self.analysis_results['training_recommendations']['frame_extraction']['frames_per_video']}
- **Extraction Method**: {self.analysis_results['training_recommendations']['frame_extraction']['extraction_method'].title()}
- **Estimated Total Frames**: {self.analysis_results['training_recommendations']['frame_extraction']['target_frame_count']:,}

## Key Findings

1. **Class Imbalance**: The dataset shows significant class imbalance with only {self.analysis_results['training_recommendations']['class_balance']['positive_ratio']:.1%} of videos containing bombus activity.

2. **Species Distribution**: Both B. vosnesenskii and B. californicus are present, with your target species (vosnesenskii) representing a subset of total bombus observations.

3. **Temporal Patterns**: Analysis of shift types and weekly patterns can inform optimal monitoring times.

4. **Site Variability**: Different sites show varying levels of bombus activity, suggesting site-specific factors affect pollinator presence.

## Next Steps for Model Development

1. **Data Preprocessing**: Extract frames from videos using the recommended strategy
2. **Data Augmentation**: Implement aggressive augmentation to address class imbalance
3. **Model Architecture**: Start with transfer learning using ResNet or EfficientNet
4. **Training Strategy**: Use class weights and careful validation splitting
5. **Evaluation**: Focus on precision/recall metrics given the imbalanced nature

## File Outputs

- `detailed_analysis_results.json`: Complete analysis results in JSON format
- `bombus_analysis_overview.png`: Visual overview of bombus observations
- `temporal_analysis.png`: Temporal patterns in data collection
- `training_recommendations.png`: ML training recommendations visualization

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_dir / "analysis_summary_report.md", 'w') as f:
            f.write(report_content)

def main():
    """Main analysis pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pollinator Dataset Analysis Tool')
    parser.add_argument('--observations', required=True, help='Path to observations Excel file')
    parser.add_argument('--output', default='analysis_output', help='Output directory for results')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = PollinatorDatasetAnalyzer(args.observations)
        
        # Generate comprehensive report
        analyzer.generate_comprehensive_report(args.output)
        
        # Print key findings
        recommendations = analyzer.analysis_results['training_recommendations']
        
        print("\n" + "="*60)
        print("POLLINATOR DATASET ANALYSIS COMPLETE")
        print("="*60)
        
        print(f"\nDataset Summary:")
        print(f"• Total videos: {recommendations['dataset_overview']['total_videos']}")
        print(f"• Videos with bombus: {recommendations['dataset_overview']['total_bombus_positive']}")
        print(f"• Positive sample ratio: {recommendations['class_balance']['positive_ratio']:.1%}")
        
        print(f"\nClass Balance Assessment: {recommendations['class_balance']['imbalance_severity'].upper()}")
        print(f"Augmentation Priority: {recommendations['augmentation_strategy']['priority'].upper()}")
        
        print(f"\nEstimated Training Data:")
        print(f"• Total frames: ~{recommendations['frame_extraction']['target_frame_count']:,}")
        print(f"• Positive frames: ~{recommendations['frame_extraction']['positive_frame_estimate']:,}")
        
        print(f"\nResults saved to: {args.output}/")
        print("• analysis_summary_report.md - Human-readable summary")
        print("• detailed_analysis_results.json - Complete analysis data")
        print("• *.png files - Visualization plots")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()