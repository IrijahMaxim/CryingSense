import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard convolutional block: Conv2d + BatchNorm2d + ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class CryingSenseCNN(nn.Module):
    """
    CNN for classifying infant cries into five categories:
    belly_pain, burp, discomfort, hunger, tired.

    Input: 4-channel 2D tensor (MFCC, Mel-spectrogram, Chroma, Delta-MFCC).
    Architecture:
      - 3 convolutional blocks (Conv2d + BatchNorm + ReLU) for hierarchical
        spectral/temporal feature extraction, each followed by MaxPool2d.
      - Global Average Pooling to produce a fixed-size feature vector.
      - Two fully connected layers integrating learned features.
      - Output: raw logits (apply softmax for probabilities at inference time).
    """

    def __init__(self, num_classes=5, in_channels=4, dropout_rate=0.3):
        super(CryingSenseCNN, self).__init__()

        # Convolutional layers: extract hierarchical spectral/temporal patterns
        # Early layers: energy bands, onsets, formant transitions
        # Deeper layers: abstract cry-type patterns (wails, bursts, harmonics)
        self.conv1 = ConvBlock(in_channels, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 64)

        # Pooling: reduces spatial dimensions, adds shift invariance, limits overfitting
        self.pool = nn.MaxPool2d(2, 2)

        # Global Average Pooling: collapses spatial dims to a fixed 64-element vector
        # regardless of variable input lengths, enabling lightweight FC integration
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers: integrate learned features into cry-type representation
        # fc1 input size is always 64 (conv3 output channels) after GAP
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional + pooling stages
        x = self.pool(self.conv1(x))   # Low-level spectral patterns
        x = self.pool(self.conv2(x))   # Mid-level feature aggregation
        x = self.pool(self.conv3(x))   # High-level cry-type representations

        # Global Average Pooling then flatten: (batch, 64, H, W) -> (batch, 64)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Dense layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)  # Raw logits; softmax is applied at inference via predict_proba()
        return x

    def predict_proba(self, x):
        """
        Run forward pass and apply softmax to produce class probabilities.
        The class with the highest probability is the predicted cry category;
        its value is the confidence score.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


if __name__ == "__main__":
    model = CryingSenseCNN()
    dummy = torch.randn(2, 4, 128, 216)  # batch, channels, features, time
    logits = model(dummy)
    print(f"Logits shape:        {logits.shape}")   # (2, 5)
    proba = model.predict_proba(dummy)
    print(f"Probabilities shape: {proba.shape}")    # (2, 5)
    print(f"Probability sums:    {proba.sum(dim=1)}")  # [1.0, 1.0]
