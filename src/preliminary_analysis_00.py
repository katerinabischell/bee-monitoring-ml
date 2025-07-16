"""
Robust Pollinator Dataset Analyzer
Author: Katerina Bischel
Project: Endangered Coastal Plant Pollinator Monitoring

This version handles missing dependencies gracefully and provides clear setup instructions.
"""

import sys
import os
from pathlib import Path

def check_and_install_requirements():
    """Check if required packages are available and provide installation instructions"""
    
    required_packages = {
        'pandas': 'pip install pandas',
        'numpy': 'pip install numpy', 
        'matplotlib': 'pip install matplotlib',
        'seaborn': 'pip install seaborn',
        'openpyxl': 'pip install openpyxl'
    }
    
    missing_packages = []
    available_packages = {}
    
    for package, install_cmd in required_packages.items():
        try:
            if package == 'pandas':
                import pandas as pd
                available_packages['pandas'] = pd
            elif package == 'numpy':
                import numpy as np
                available_packages['numpy'] = np
            elif package == 'matplotlib':
                import matplotlib.pyplot as plt
                available_packages['matplotlib'] = plt
            elif package == 'seaborn':
                import seaborn as sns
                available_packages['seaborn'] = sns
            elif package == 'openpyxl':
                import openpyxl
                available_packages['openpyxl'] = openpyxl
                
            print(f"✅ {package} is available")
            
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append((package, install_cmd))
    
    if missing_packages:
        print(f"\n🔧 To install missing packages, run these commands in your terminal:")
        print(f"cd Desktop/bee-monitoring-ml")
        for package, cmd in missing_packages:
            print(f"{cmd}")
        
        print(f"\nOr install all at once:")
        print(f"pip install pandas numpy matplotlib seaborn openpyxl")
        
        return False, None
    
    return True, available_packages

def analyze_pollinator_dataset_simple(excel_file_path, output_folder="analysis_output"):
    """
    Basic analysis that works even with minimal dependencies
    """
    
    print(f"🐝 Starting Basic Pollinator Dataset Analysis...")
    
    # Check if we can proceed
    packages_ok, packages = check_and_install_requirements()
    
    if not packages_ok:
        print(f"\n⚠️  Please install the missing packages first, then run this script again.")
        return None
    
    # Now we know packages are available
    pd = packages['pandas']
    np = packages['numpy'] 
    plt = packages['matplotlib']
    
    # Create output directory
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the Excel file
        print(f"📊 Loading data from: {excel_file_path}")
        
        if not Path(excel_file_path).exists():
            print(f"❌ Excel file not found: {excel_file_path}")
            print(f"📁 Current directory: {os.getcwd()}")
            print(f"📁 Files in current directory:")
            for file in os.listdir("."):
                if file.endswith(('.xlsx', '.xls')):
                    print(f"   📄 {file}")
            return None
        
        # Read both sheets
        birds_beak_df = pd.read_excel(excel_file_path, sheet_name='Birds Beak')
        milkvetch_df = pd.read_excel(excel_file_path, sheet_name='Ventura Milkvetch')
        
        print(f"✅ Loaded {len(birds_beak_df)} Birds Beak observations")
        print(f"✅ Loaded {len(milkvetch_df)} Ventura Milkvetch observations")
        
        # Basic analysis without complex dependencies
        results = analyze_bombus_activity(birds_beak_df, milkvetch_df)
        
        # Create simple visualizations
        if 'seaborn' in packages:
            create_basic_plots(results, output_dir, plt, packages['seaborn'])
        else:
            create_matplotlib_plots(results, output_dir, plt)
        
        # Save results
        save_results(results, output_dir)
        
        # Print summary
        print_summary(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(f"💡 Make sure your Excel file is not open in another program")
        return None

def analyze_bombus_activity(birds_beak_df, milkvetch_df):
    """Analyze bombus activity in the dataset"""
    
    results = {}
    
    for plant_type, df in [("Birds Beak", birds_beak_df), ("Ventura Milkvetch", milkvetch_df)]:
        
        # Clean the data
        df = df.copy()
        df['Activity on site'] = df['Activity on site'].fillna('none').astype(str).str.lower()
        df['Activity around '] = df['Activity around '].fillna('none').astype(str).str.lower()
        
        # Define bombus keywords
        bombus_keywords = ['bombus', 'b. californicus', 'b. vosnesenskii', 'bombus v', 'bombus vos']
        
        # Find bombus observations
        bombus_pattern = '|'.join(bombus_keywords)
        bombus_on_site = df['Activity on site'].str.contains(bombus_pattern, na=False)
        bombus_around = df['Activity around '].str.contains(bombus_pattern, na=False)
        bombus_any = bombus_on_site | bombus_around
        
        # Species-specific counts
        vos_pattern = 'vosnesenskii|bombus v|bombus vos'
        cal_pattern = 'californicus'
        
        vos_count = (df['Activity on site'].str.contains(vos_pattern, na=False) | 
                    df['Activity around '].str.contains(vos_pattern, na=False)).sum()
        cal_count = (df['Activity on site'].str.contains(cal_pattern, na=False) | 
                    df['Activity around '].str.contains(cal_pattern, na=False)).sum()
        
        # Store results
        plant_results = {
            'total_videos': len(df),
            'bombus_videos': bombus_any.sum(),
            'bombus_percentage': (bombus_any.sum() / len(df)) * 100 if len(df) > 0 else 0,
            'vosnesenskii_count': vos_count,
            'californicus_count': cal_count,
        }
        
        # Add temporal analysis if data is available
        if bombus_any.sum() > 0:
            bombus_subset = df[bombus_any]
            if 'Shift type' in bombus_subset.columns:
                plant_results['bombus_by_shift'] = bombus_subset['Shift type'].value_counts().to_dict()
            if 'Site #' in bombus_subset.columns:
                plant_results['bombus_by_site'] = bombus_subset['Site #'].value_counts().to_dict()
        
        results[plant_type] = plant_results
    
    return results

def create_basic_plots(results, output_dir, plt, sns=None):
    """Create basic visualization plots"""
    
    if sns:
        plt.style.use('seaborn-v0_8')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pollinator Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Data for plots
    plant_types = ['Birds Beak', 'Ventura Milkvetch']
    bombus_counts = [results['Birds Beak']['bombus_videos'], results['Ventura Milkvetch']['bombus_videos']]
    total_counts = [results['Birds Beak']['total_videos'], results['Ventura Milkvetch']['total_videos']]
    detection_rates = [results['Birds Beak']['bombus_percentage'], results['Ventura Milkvetch']['bombus_percentage']]
    
    # 1. Bombus detections by plant type
    bars1 = axes[0, 0].bar(plant_types, bombus_counts, color=['skyblue', 'lightcoral'], alpha=0.7)
    axes[0, 0].set_title('Videos with Bombus Activity')
    axes[0, 0].set_ylabel('Number of Videos')
    
    # Add value labels
    for bar, count in zip(bars1, bombus_counts):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       str(count), ha='center', fontweight='bold')
    
    # 2. Detection rates
    bars2 = axes[0, 1].bar(plant_types, detection_rates, color=['lightgreen', 'gold'], alpha=0.7)
    axes[0, 1].set_title('Bombus Detection Rates')
    axes[0, 1].set_ylabel('Percentage of Videos')
    
    # Add percentage labels
    for bar, rate in zip(bars2, detection_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{rate:.1f}%', ha='center', fontweight='bold')
    
    # 3. Species comparison
    vos_counts = [results['Birds Beak']['vosnesenskii_count'], results['Ventura Milkvetch']['vosnesenskii_count']]
    cal_counts = [results['Birds Beak']['californicus_count'], results['Ventura Milkvetch']['californicus_count']]
    
    x = range(len(plant_types))
    width = 0.35
    
    axes[1, 0].bar([i - width/2 for i in x], vos_counts, width, label='B. vosnesenskii', 
                   color='steelblue', alpha=0.7)
    axes[1, 0].bar([i + width/2 for i in x], cal_counts, width, label='B. californicus', 
                   color='darkorange', alpha=0.7)
    axes[1, 0].set_title('Bombus Species Observations')
    axes[1, 0].set_ylabel('Number of Observations')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(plant_types)
    axes[1, 0].legend()
    
    # 4. Overall dataset composition
    total_bombus = sum(bombus_counts)
    total_videos = sum(total_counts)
    no_bombus = total_videos - total_bombus
    
    labels = ['Videos with Bombus', 'Videos without Bombus']
    sizes = [total_bombus, no_bombus]
    colors = ['lightgreen', 'lightcoral']
    
    axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 1].set_title('Overall Dataset Composition')
    
    plt.tight_layout()
    plt.savefig(output_dir / "pollinator_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved to: {output_dir}/pollinator_analysis.png")

def create_matplotlib_plots(results, output_dir, plt):
    """Create plots using only matplotlib (fallback)"""
    create_basic_plots(results, output_dir, plt, sns=None)

def save_results(results, output_dir):
    """Save results to files"""
    
    # Create a simple text summary
    summary_text = f"""POLLINATOR DATASET ANALYSIS SUMMARY
{'='*50}

BIRDS BEAK (Chloropyron maritimum):
- Total videos: {results['Birds Beak']['total_videos']}
- Videos with bombus: {results['Birds Beak']['bombus_videos']}
- Detection rate: {results['Birds Beak']['bombus_percentage']:.1f}%
- B. vosnesenskii observations: {results['Birds Beak']['vosnesenskii_count']}
- B. californicus observations: {results['Birds Beak']['californicus_count']}

VENTURA MILKVETCH (Astragalus pycnostachyus):
- Total videos: {results['Ventura Milkvetch']['total_videos']}
- Videos with bombus: {results['Ventura Milkvetch']['bombus_videos']}
- Detection rate: {results['Ventura Milkvetch']['bombus_percentage']:.1f}%
- B. vosnesenskii observations: {results['Ventura Milkvetch']['vosnesenskii_count']}
- B. californicus observations: {results['Ventura Milkvetch']['californicus_count']}

OVERALL STATISTICS:
- Total videos: {results['Birds Beak']['total_videos'] + results['Ventura Milkvetch']['total_videos']}
- Total videos with bombus: {results['Birds Beak']['bombus_videos'] + results['Ventura Milkvetch']['bombus_videos']}
- Overall detection rate: {((results['Birds Beak']['bombus_videos'] + results['Ventura Milkvetch']['bombus_videos']) / (results['Birds Beak']['total_videos'] + results['Ventura Milkvetch']['total_videos'])) * 100:.1f}%

MACHINE LEARNING RECOMMENDATIONS:
- Class imbalance: {"SEVERE" if ((results['Birds Beak']['bombus_videos'] + results['Ventura Milkvetch']['bombus_videos']) / (results['Birds Beak']['total_videos'] + results['Ventura Milkvetch']['total_videos'])) < 0.1 else "MODERATE"}
- Data augmentation: HIGH PRIORITY
- Recommended batch size: {min(32, max(8, (results['Birds Beak']['bombus_videos'] + results['Ventura Milkvetch']['bombus_videos']) // 10))}
- Estimated frames needed: ~{(results['Birds Beak']['total_videos'] + results['Ventura Milkvetch']['total_videos']) * 50:,}
"""
    
    with open(output_dir / "analysis_summary.txt", 'w') as f:
        f.write(summary_text)
    
    print(f"📄 Summary saved to: {output_dir}/analysis_summary.txt")

def print_summary(results):
    """Print analysis summary to console"""
    
    total_videos = results['Birds Beak']['total_videos'] + results['Ventura Milkvetch']['total_videos']
    total_bombus = results['Birds Beak']['bombus_videos'] + results['Ventura Milkvetch']['bombus_videos']
    overall_rate = (total_bombus / total_videos) * 100 if total_videos > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"🎯 ANALYSIS COMPLETE - KEY FINDINGS")
    print(f"="*60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total videos: {total_videos}")
    print(f"   Videos with bombus activity: {total_bombus}")
    print(f"   Overall detection rate: {overall_rate:.1f}%")
    
    print(f"\n🐝 Species Breakdown:")
    total_vos = results['Birds Beak']['vosnesenskii_count'] + results['Ventura Milkvetch']['vosnesenskii_count']
    total_cal = results['Birds Beak']['californicus_count'] + results['Ventura Milkvetch']['californicus_count']
    print(f"   B. vosnesenskii (target): {total_vos} observations")
    print(f"   B. californicus: {total_cal} observations")
    
    print(f"\n🤖 ML Recommendations:")
    positive_ratio = total_bombus / total_videos if total_videos > 0 else 0
    
    if positive_ratio < 0.1:
        imbalance = "SEVERE"
    elif positive_ratio < 0.3:
        imbalance = "MODERATE" 
    else:
        imbalance = "BALANCED"
    
    print(f"   Class imbalance: {imbalance} ({positive_ratio:.1%} positive)")
    print(f"   Recommended approach: Transfer learning + data augmentation")
    print(f"   Estimated training frames: ~{total_videos * 50:,}")
    
    if positive_ratio < 0.2:
        print(f"\n⚠️  Important: Your dataset has significant class imbalance.")
        print(f"   Consider aggressive data augmentation and class weighting.")

# Main execution
if __name__ == "__main__":
    
    print(f"🚀 Pollinator Dataset Analyzer")
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Look for Excel file
    excel_files = [f for f in os.listdir(".") if f.endswith(('.xlsx', '.xls'))]
    
    if 'Collection Observations3.xlsx' in excel_files:
        excel_file = 'Collection Observations3.xlsx'
    elif len(excel_files) == 1:
        excel_file = excel_files[0]
        print(f"📄 Found Excel file: {excel_file}")
    elif len(excel_files) > 1:
        print(f"📄 Multiple Excel files found:")
        for i, file in enumerate(excel_files):
            print(f"   {i+1}. {file}")
        print(f"📝 Using: Collection Observations3.xlsx (default)")
        excel_file = 'Collection Observations3.xlsx' if 'Collection Observations3.xlsx' in excel_files else excel_files[0]
    else:
        print(f"❌ No Excel files found in current directory")
        print(f"📁 Make sure 'Collection Observations3.xlsx' is in this folder")
        exit(1)
    
    # Run analysis
    results = analyze_pollinator_dataset_simple(excel_file)
    
    if results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Check the 'analysis_output' folder for detailed results")