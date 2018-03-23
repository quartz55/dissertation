import torch.nn as nn

class Mnist4Layers(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
      # Layer 1
      nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=1),
      nn.Dropout(p=0.5),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      # Layer 2
      nn.Conv2d(16, 32, 3, padding=1),
      nn.Dropout(0.5),
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2),

      # Layer 3
      nn.Conv2d(32, 64, 3, padding=1),
      nn.Dropout(0.5),
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2),

      # Layer 4
      nn.Conv2d(64, 128, 3, padding=1),
      nn.Dropout(0.5),
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2)
    )
    self.fc = nn.Linear(128, 10)
    
  def forward(self, x):
    return self.fc(self.conv(x).squeeze())