"""
Audio Visualization Script for CryingSense

Generates comprehensive visualizations and statistics for the dataset:

Visualization Types:
  1. Audio Features (per file):
     - Waveform (time-domain amplitude)
     - Spectrogram (time-frequency representation)
     - Mel Spectrogram (perceptually-weighted frequencies)
     - MFCC (Mel-Frequency Cepstral Coefficients)
     - Chroma Features (pitch class representation)
     
  2. Dataset Statistics:
     - Class distribution (sample counts per category)
     - Audio duration statistics
     - Feature distribution plots
     - Summary reports

Usage:
  python visualize_dataset.py                    # Visualize sample files from each category
  python visualize_dataset.py --all              # Visualize ALL audio files (slow!)
  python visualize_dataset.py --stats-only       # Generate only dataset statistics
  python visualize_dataset.py --samples 5        # Visualize 5 random samples per category

Saves visualizations to: dataset/visualizations/
"""

import os
import sys
import argparse
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def collect_dataset_stats(input_dir, sr=16000):
    """
    Collect statistics about the dataset.
    
    Args:
        input_dir: Directory containing audio files
        sr: Sample rate
        
    Returns:
        dict: Statistics about files, durations, and categories
    """
    stats = {
        'categories': defaultdict(lambda: {'count': 0, 'durations': []}),
        'total_files': 0,
        'total_duration': 0
    }
    
    print("\nCollecting dataset statistics...")
    
    for root, _, files in os.walk(input_dir):
        category = os.path.basename(root)
        wav_files = [f for f in files if f.endswith('.wav')]
        
        if not wav_files:
            continue
            
        for file in tqdm(wav_files, desc=f"Analyzing {category}"):
            file_path = os.path.join(root, file)
            try:
                # Get audio duration without loading full file (faster)
                duration = librosa.get_duration(path=file_path)
                stats['categories'][category]['count'] += 1
                stats['categories'][category]['durations'].append(duration)
                stats['total_files'] += 1
                stats['total_duration'] += duration
            except Exception as e:
                print(f"\nWarning: Could not analyze {file_path}: {str(e)}")
    
    return stats


def plot_dataset_statistics(stats, output_dir):
    """
    Create statistical visualization plots for the dataset.
    
    Args:
        stats: Dictionary with dataset statistics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    categories = list(stats['categories'].keys())
    counts = [stats['categories'][cat]['count'] for cat in categories]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Class Distribution (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    bars = ax1.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Class Distribution (Pie Chart)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors,
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Dataset Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    # 3. Duration Statistics (Box Plot)
    ax3 = fig.add_subplot(gs[1, 0])
    durations_list = [stats['categories'][cat]['durations'] for cat in categories]
    bp = ax3.boxplot(durations_list, labels=categories, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    ax3.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Audio Duration Distribution by Category', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Summary Statistics (Text)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_text = "Dataset Summary\n" + "="*40 + "\n\n"
    summary_text += f"Total Files: {stats['total_files']}\n"
    summary_text += f"Total Duration: {stats['total_duration']:.2f} seconds ({stats['total_duration']/60:.2f} minutes)\n\n"
    summary_text += "Per Category:\n" + "-"*40 + "\n"
    
    for cat in sorted(categories):
        cat_stats = stats['categories'][cat]
        durations = cat_stats['durations']
        summary_text += f"\n{cat.upper()}:\n"
        summary_text += f"  Files: {cat_stats['count']}\n"
        summary_text += f"  Total Duration: {sum(durations):.2f}s\n"
        summary_text += f"  Avg Duration: {np.mean(durations):.2f}s\n"
        summary_text += f"  Min Duration: {np.min(durations):.2f}s\n"
        summary_text += f"  Max Duration: {np.max(durations):.2f}s\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    output_path = os.path.join(output_dir, 'dataset_statistics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n✓ Dataset statistics saved to: {output_path}")
    
    # Also save text summary
    text_output_path = os.path.join(output_dir, 'dataset_summary.txt')
    with open(text_output_path, 'w') as f:
        f.write(summary_text)
    print(f"✓ Text summary saved to: {text_output_path}")


def visualize_audio(audio_path, output_dir, sr=16000):
    """
    Create comprehensive visualizations for a single audio file.
    
    Args:
        audio_path: Path to .wav file
        output_dir: Directory to save visualizations
        sr: Sample rate (default: 16000)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a large figure with 5 subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Waveform
        ax1 = plt.subplot(5, 1, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.8, color='#00BFFF')
        ax1.set_title(f'Waveform - {filename}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # 2. Spectrogram
        ax2 = plt.subplot(5, 1, 2)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='viridis')
        ax2.set_title('Spectrogram', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Frequency (Hz)')
        plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        # 3. Mel Spectrogram
        ax3 = plt.subplot(5, 1, 3)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax3, cmap='magma')
        ax3.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Mel Frequency')
        plt.colorbar(img, ax=ax3, format='%+2.0f dB')
        
        # 4. MFCC
        ax4 = plt.subplot(5, 1, 4)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax4, cmap='coolwarm')
        ax4.set_title('MFCC (Mel-Frequency Cepstral Coefficients)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('MFCC Coefficients')
        plt.colorbar(img, ax=ax4)
        
        # 5. Chroma Features
        ax5 = plt.subplot(5, 1, 5)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax5, cmap='plasma')
        ax5.set_title('Chroma Features (Pitch Class Representation)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Pitch Class')
        ax5.set_xlabel('Time (s)')
        plt.colorbar(img, ax=ax5)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"{filename}_visualization.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"\nError visualizing {audio_path}: {str(e)}")
        return False


def select_samples(input_dir, samples_per_category=3):
    """
    Select random samples from each category.
    
    Args:
        input_dir: Root directory containing audio files
        samples_per_category: Number of samples to select per category
        
    Returns:
        list: List of selected file paths
    """
    samples = []
    
    for root, _, files in os.walk(input_dir):
        wav_files = [os.path.join(root, f) for f in files if f.endswith('.wav')]
        
        if not wav_files:
            continue
            
        # Randomly select samples
        num_to_select = min(samples_per_category, len(wav_files))
        selected = random.sample(wav_files, num_to_select)
        samples.extend(selected)
    
    return samples


def visualize_dataset(input_dir, output_dir, sr=160000, all_files=False, samples_per_category=3):
    """
    Visualize audio files in the dataset.
    
    Args:
        input_dir: Root directory containing audio files
        output_dir: Root directory for visualizations
        sr: Sample rate (default: 16000)
        all_files: If True, visualize all files; if False, sample files
        samples_per_category: Number of samples per category (if not all_files)
    
    Returns:
        dict: Statistics about the visualization process
    """
    stats = {
        'total_files': 0,
        'visualized_files': 0,
        'errors': []
    }
    
    # Determine which files to visualize
    if all_files:
        print("\n⚠️  Visualizing ALL audio files (this may take a while)...")
        files_to_visualize = []
        for root, _, files in os.walk(input_dir):
            files_to_visualize.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])
    else:
        print(f"\nSelecting {samples_per_category} random samples per category...")
        files_to_visualize = select_samples(input_dir, samples_per_category)
    
    print(f"Total files to visualize: {len(files_to_visualize)}\n")
    
    # Visualize selected files
    for file_path in tqdm(files_to_visualize, desc="Visualizing audio files"):
        stats['total_files'] += 1
        
        # Get category and create output directory
        rel_path = os.path.relpath(file_path, input_dir)
        category = os.path.dirname(rel_path)
        if category:
            current_output_dir = os.path.join(output_dir, category)
        else:
            current_output_dir = output_dir
        
        if visualize_audio(file_path, current_output_dir, sr=sr):
            stats['visualized_files'] += 1
        else:
            stats['errors'].append(file_path)
    
    return stats


def main():
    """Main function to run visualization pipeline."""
    parser = argparse.ArgumentParser(
        description='Visualize CryingSense audio dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize 3 random samples per category (default)
  python visualize_dataset.py
  
  # Visualize all audio files
  python visualize_dataset.py --all
  
  # Visualize 5 random samples per category
  python visualize_dataset.py --samples 5
  
  # Generate only dataset statistics
  python visualize_dataset.py --stats-only
  
  # Use raw audio instead of cleaned
  python visualize_dataset.py --raw
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Visualize ALL audio files (slow!)')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of samples per category (default: 3)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Generate only dataset statistics')
    parser.add_argument('--raw', action='store_true',
                       help='Use raw audio instead of cleaned')
    
    args = parser.parse_args()
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define directories
    if args.raw:
        input_dir = os.path.join(project_root, "dataset", "raw")
        output_dir = os.path.join(project_root, "dataset", "visualizations", "raw")
        dataset_name = "RAW"
    else:
        input_dir = os.path.join(project_root, "dataset", "processed", "cleaned")
        output_dir = os.path.join(project_root, "dataset", "visualizations", "cleaned")
        dataset_name = "CLEANED"
    
    print("=" * 70)
    print(f"CryingSense Audio Visualization - {dataset_name}")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"\n❌ Error: Input directory does not exist: {input_dir}")
        if not args.raw:
            print("\nPlease run preprocessing first:")
            print("  python preprocess_audio.py")
        sys.exit(1)
    
    # Collect and visualize statistics
    stats_data = collect_dataset_stats(input_dir)
    plot_dataset_statistics(stats_data, output_dir)
    
    # Visualize audio files (unless stats-only mode)
    if not args.stats_only:
        print()
        print("Generating audio visualizations:")
        print("  ✓ Waveform")
        print("  ✓ Spectrogram")
        print("  ✓ Mel Spectrogram")
        print("  ✓ MFCC")
        print("  ✓ Chroma Features")
        
        viz_stats = visualize_dataset(input_dir, output_dir, 
                                      all_files=args.all, 
                                      samples_per_category=args.samples)
        
        print()
        print("=" * 70)
        print("Visualization Complete")
        print("=" * 70)
        print(f"Total files visualized: {viz_stats['visualized_files']}/{viz_stats['total_files']}")
        print(f"Errors: {len(viz_stats['errors'])}")
        
        if viz_stats['errors']:
            print("\n❌ Errors encountered:")
            for error_file in viz_stats['errors'][:10]:
                print(f"  - {error_file}")
            if len(viz_stats['errors']) > 10:
                print(f"  ... and {len(viz_stats['errors']) - 10} more")
        
        if viz_stats['visualized_files'] > 0:
            print()
            print(f"✓ Visualizations saved to: {output_dir}")
    
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
