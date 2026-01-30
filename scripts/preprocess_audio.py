
import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import noisereduce as nr
import random


def add_background_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return augmented.astype(y.dtype)

def time_stretch(y, rate=1.0):
    return librosa.effects.time_stretch(y, rate)

def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr, n_steps)

def preprocess_audio(input_dir, output_dir, sample_rate=16000, duration=5.0, augment=True):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            if not file.endswith('.wav'):
                continue
            file_path = os.path.join(root, file)
            y, sr = librosa.load(file_path, sr=sample_rate)
            # Trim silence
            y, _ = librosa.effects.trim(y)
            # Noise reduction
            y = nr.reduce_noise(y=y, sr=sr)
            # Normalize
            y = librosa.util.normalize(y)
            # Pad or trim to fixed duration
            target_length = int(sample_rate * duration)
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)))
            # Save processed audio
            rel_path = os.path.relpath(file_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            wavfile.write(out_path, sample_rate, (y * 32767).astype(np.int16))

            # Data augmentation
            if augment:
                # Time stretch (random between 0.8 and 1.2)
                ts_rate = random.uniform(0.8, 1.2)
                y_ts = time_stretch(y, ts_rate)
                if len(y_ts) > target_length:
                    y_ts = y_ts[:target_length]
                else:
                    y_ts = np.pad(y_ts, (0, target_length - len(y_ts)))
                ts_path = out_path.replace('.wav', f'_timestretch.wav')
                wavfile.write(ts_path, sample_rate, (y_ts * 32767).astype(np.int16))

                # Pitch shift (random between -2 and +2 semitones)
                n_steps = random.uniform(-2, 2)
                y_ps = pitch_shift(y, sr, n_steps)
                if len(y_ps) > target_length:
                    y_ps = y_ps[:target_length]
                else:
                    y_ps = np.pad(y_ps, (0, target_length - len(y_ps)))
                ps_path = out_path.replace('.wav', f'_pitchshift.wav')
                wavfile.write(ps_path, sample_rate, (y_ps * 32767).astype(np.int16))

                # Add background noise
                y_bn = add_background_noise(y)
                if len(y_bn) > target_length:
                    y_bn = y_bn[:target_length]
                else:
                    y_bn = np.pad(y_bn, (0, target_length - len(y_bn)))
                bn_path = out_path.replace('.wav', f'_noise.wav')
                wavfile.write(bn_path, sample_rate, (y_bn * 32767).astype(np.int16))

if __name__ == "__main__":
    preprocess_audio("../dataset/raw", "../dataset/processed/cleaned", augment=True)
