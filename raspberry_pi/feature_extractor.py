"""
Feature Extraction — Raspberry Pi 3B+ Pipeline

Builds the 4-channel input tensor expected by CryingSenseCNN:
  Channel 0 — MFCC          (padded to 128 × T)
  Channel 1 — Mel spectrogram (dB scale, 128 × T)
  Channel 2 — Chroma        (padded to 128 × T)
  Channel 3 — Δ-MFCC        (padded to 128 × T)

All operations use float32 — safe for 1 GB RAM at 16 kHz / 5 s audio.
"""

import numpy as np
import librosa

from config import (
    SAMPLE_RATE,
    DURATION,
    N_MFCC,
    N_MELS,
    N_CHROMA,
    N_FFT,
    HOP_LENGTH,
)


class FeatureExtractor:
    """Mel-spectrogram-based feature builder for CryingSenseCNN."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        duration: float = DURATION,
        n_mfcc: int = N_MFCC,
        n_mels: int = N_MELS,
        n_chroma: int = N_CHROMA,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ):
        self.sr = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Target dimensions — must stay in sync with training
        self.time_steps = int(np.ceil((sample_rate * duration) / hop_length))
        self.height = max(n_mfcc, n_mels, n_chroma)  # 128

    # ── public API ───────────────────────────────────────────────────────

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        audio : 1-D float32 numpy array (already preprocessed).

        Returns
        -------
        (4, height, time_steps)  float32 numpy array.
        """
        mfcc = self._mfcc(audio)
        mel = self._mel(audio)
        chroma = self._chroma(audio)
        delta_mfcc = self._delta_mfcc(mfcc)

        return np.stack(
            [
                self._fit(mfcc),
                self._fit(mel),
                self._fit(chroma),
                self._fit(delta_mfcc),
            ],
            axis=0,
        ).astype(np.float32)

    def get_shape(self) -> tuple:
        return (4, self.height, self.time_steps)

    # ── feature helpers ──────────────────────────────────────────────────

    def _mfcc(self, audio: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def _mel(self, audio: np.ndarray) -> np.ndarray:
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return librosa.power_to_db(S, ref=np.max)

    def _chroma(self, audio: np.ndarray) -> np.ndarray:
        return librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_chroma=self.n_chroma,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    @staticmethod
    def _delta_mfcc(mfcc: np.ndarray) -> np.ndarray:
        delta = np.zeros_like(mfcc)
        delta[:, 1:] = mfcc[:, 1:] - mfcc[:, :-1]
        return delta

    def _fit(self, feat: np.ndarray) -> np.ndarray:
        """Pad / crop to (height, time_steps)."""
        out = np.zeros((self.height, self.time_steps), dtype=feat.dtype)
        h = min(feat.shape[0], self.height)
        w = min(feat.shape[1], self.time_steps)
        out[:h, :w] = feat[:h, :w]
        return out
