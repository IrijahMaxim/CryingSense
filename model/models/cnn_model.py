import torch
import torch.nn as nn
import torch.nn.functional as F


# Depthwise Separable Conv2d block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class CryingSenseCNN(nn.Module):
    def __init__(self, num_classes=5, in_channels=4, dropout_rate=0.3, use_gap=True):
        super(CryingSenseCNN, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, 16)
        self.conv2 = DepthwiseSeparableConv(16, 32)
        self.conv3 = DepthwiseSeparableConv(32, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.use_gap = use_gap
        self.dropout = nn.Dropout(dropout_rate)
        # fc1 and flattening will be set after seeing input shape
        self._fc1 = None
        self.fc2 = nn.Linear(128, num_classes)

    def _get_flattened_size(self, x):
        # Pass a dummy tensor through conv/pool to get flattened size
        with torch.no_grad():
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.pool(x)
            if self.use_gap:
                x = F.adaptive_avg_pool2d(x, (1, 1))
                return x.shape[1]
            else:
                return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        if self.use_gap:
            x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, channels, 1, 1)
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)
        # Lazy init fc1
        if self._fc1 is None:
            self._fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        x = self.dropout(F.relu(self._fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Test model shape for both flatten and GAP
    for use_gap in [True, False]:
        print(f"Testing with use_gap={use_gap}")
        model = CryingSenseCNN(use_gap=use_gap)
        dummy = torch.randn(2, 4, 128, 216)  # batch, channels, features, time
        out = model(dummy)
        print(out.shape)  # Should be (2, 5)
