import os
import numpy as np
import librosa
from tqdm import tqdm


def spec_augment(mel_spectrogram, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.15):
    spec = mel_spectrogram.copy()
    num_mel_channels = spec.shape[0]
    num_time_steps = spec.shape[1]
    for _ in range(num_mask):
        # Frequency masking
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
        num_freqs_to_mask = int(freq_percentage * num_mel_channels)
        f0 = np.random.randint(0, num_mel_channels - num_freqs_to_mask)
        spec[f0:f0 + num_freqs_to_mask, :] = 0
        # Time masking
        time_percentage = np.random.uniform(0.0, time_masking_max_percentage)
        num_times_to_mask = int(time_percentage * num_time_steps)
        t0 = np.random.randint(0, num_time_steps - num_times_to_mask)
        spec[:, t0:t0 + num_times_to_mask] = 0
    return spec

def pad_or_crop(feature, target_shape):
    # feature: (n_features, time)
    padded = np.zeros(target_shape, dtype=feature.dtype)
    min_shape = (min(feature.shape[0], target_shape[0]), min(feature.shape[1], target_shape[1]))
    padded[:min_shape[0], :min_shape[1]] = feature[:min_shape[0], :min_shape[1]]
    return padded

def extract_features(input_dir, output_dir, sample_rate=16000, n_mfcc=40, n_mels=128, n_chroma=12, duration=5.0, target_time_steps=216):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            if not file.endswith('.wav'):
                continue
            file_path = os.path.join(root, file)
            y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc = pad_or_crop(mfcc, (n_mfcc, target_time_steps))
            # Mel Spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = pad_or_crop(mel_db, (n_mels, target_time_steps))
            # SpecAugment
            mel_db_aug = spec_augment(mel_db)
            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
            chroma = pad_or_crop(chroma, (n_chroma, target_time_steps))
            # Stack features for model input: (channels, features, time)
            features = np.stack([mfcc, mel_db, mel_db_aug, chroma], axis=0)
            rel_path = os.path.splitext(os.path.relpath(file_path, input_dir))[0] + '.npy'
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, features)

if __name__ == "__main__":
    extract_features("../dataset/processed/cleaned", "../dataset/processed/features")
