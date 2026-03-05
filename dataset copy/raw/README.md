# Speech Audio Folder

This folder contains **human speech audio samples** that will be treated as **noise** by the CryingSense model.

## Purpose

The model needs to distinguish baby cries from other sounds, including:
- Adult speech (parents talking)
- Baby babbling/cooing (non-crying vocalizations)
- Children talking
- TV/radio speech
- Phone conversations

By training on speech samples labeled as "noise", the model learns to **ignore speech** and focus only on crying patterns.

## Audio Requirements

### Format
- ✅ `.wav` files
- ✅ Sample rate: Any (will be resampled to 16 kHz)
- ✅ Duration: Any (will be trimmed/padded to 5 seconds)

### Content Guidelines

**Include:**
- ✅ Adult speech (male and female voices)
- ✅ Baby babbling, cooing, laughing (non-crying)
- ✅ Children talking
- ✅ Conversations (multiple speakers)
- ✅ TV/radio speech
- ✅ Singing, humming
- ✅ Various languages
- ✅ Different volumes and distances

**Avoid:**
- ❌ Baby crying sounds (these belong in other categories)
- ❌ Music without vocals
- ❌ Pure environmental sounds

### How to Add Speech Samples

1. **Place `.wav` files directly in this folder:**
   ```
   dataset/raw/speech/
   ├── adult_conversation_01.wav
   ├── baby_babbling_01.wav
   ├── tv_speech_sample.wav
   └── ...
   ```

2. **Run preprocessing:**
   ```bash
   python scripts/preprocess_audio.py
   ```
   
   This will clean and augment the speech samples.

3. **Extract features:**
   ```bash
   python scripts/feature_extraction.py
   ```

4. **Train/retrain the model:**
   ```bash
   python model/training/train.py
   ```

## Sources for Speech Data

### Free Datasets
- **LibriSpeech** - Clean English speech recordings
- **Common Voice (Mozilla)** - Multi-language speech
- **VoxCeleb** - Celebrity speech samples
- **TIMIT** - Phonetically diverse speech

### Recording Your Own
- Use a phone or microphone
- Record family conversations (with permission)
- Include baby babbling/cooing sounds
- Vary the distance from microphone
- Include background TV/radio

## Processing Pipeline

```
speech/*.wav
    ↓ (preprocess_audio.py)
dataset/processed/cleaned/speech/*.wav
    ↓ (feature_extraction.py)
dataset/processed/features/*/speech/*.npy
    ↓ (train.py)
Model learns to classify as "noise" (ignore)
```

## Model Behavior

When trained with speech samples:
- **Speech input** → Model predicts "noise" → ❌ No alert
- **Baby crying** → Model predicts cry type → ✅ Alert triggered
- **Mixed (cry + speech)** → Model focuses on cry → ✅ Alert triggered

This ensures the system only responds to actual crying, not parents talking!

## Current Status

Check file count:
```bash
ls dataset/raw/speech/ | Measure-Object
```

## Notes

- More diverse speech samples = better model performance
- Aim for at least 50-100 speech samples
- Balance with cry samples (don't overwhelm with speech data)
- The model treats speech the same as other environmental noise
