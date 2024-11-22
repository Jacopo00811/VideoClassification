from torch import nn

class TheConvolver3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Convolution 1
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Convolution 2
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Convolution 3
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        
        # Input size: (3, 10, 64, 64)
        # After Conv and Pooling: (64, 1, 8, 8)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 1 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
