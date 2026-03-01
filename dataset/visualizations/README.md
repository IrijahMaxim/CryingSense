# Feature Visualizations

This folder contains visual representations of extracted audio features saved as image files (JPG/PNG).

## Contents

- **spectrograms/**: Spectrogram visualizations of audio signals
- **mel_spectrograms/**: Mel-scale spectrogram visualizations
- **mfcc/**: MFCC (Mel-Frequency Cepstral Coefficients) visualizations
- **waveforms/**: Time-domain waveform plots
- **other/**: Additional feature visualizations

## Sampling Strategy

To keep visualization counts manageable, the feature extraction script automatically:
- Randomly selects ~100 samples total for visualization
- Distributes samples evenly across all non-noise classes
- Excludes the 'noise' category from visualizations
- Processes ALL audio files for features, but only visualizes the selected subset

This provides representative examples while keeping storage requirements reasonable.

## Purpose

These visualizations are useful for:
- Visual inspection of extracted features
- Quality control and data validation
- Presentation and documentation
- Model interpretation and debugging

## File Naming Convention

```
{category}_{filename}_{feature_type}.png
```

Example: `hunger_baby_cry_001_mel_spectrogram.png`
