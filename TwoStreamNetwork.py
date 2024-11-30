import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from datasets import FrameImageDataset, FrameVideoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class StreamCNN(nn.Module):
    def __init__(self, hyperparameters, in_channels, load_pretrained=True):
        super(StreamCNN, self).__init__()
        self.num_classes = hyperparameters['num_classes']
        
        self.load(load_pretrained)
                
        self.features = nn.Sequential(
        # conv1
        nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # conv2
        
        nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # conv3
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        
        # conv4
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        
        # conv5
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        self.classifier = nn.Sequential(
            # fc6
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            # fc7
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            # Apply softmax
            nn.Linear(2048, self.num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
                
    def load(self, load_pretrained):
        if load_pretrained:
            self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            self.model = models.resnet152(weights=None)



class TwoStream(nn.Module):
    def __init__(self, hyperparameters, timesteps, load_pretrained=True):
        super(TwoStream, self).__init__()
        self.num_classes = hyperparameters['num_classes']
        
        # self.load(load_pretrained)
        
        # Spatial Stream
        self.spatial_stream = StreamCNN(hyperparameters, in_channels=3, load_pretrained=load_pretrained)
        
        # Temporal Stream
        self.temporal_stream = StreamCNN(hyperparameters, in_channels=(2*(timesteps-1)), load_pretrained=load_pretrained)
        
        # Fusion Layer
        # self.fusion_layer = nn.Linear(2*self.num_classes, self.num_classes)
        feature_size = 512 * 7 * 7
        self.fusion_layer = nn.Linear(feature_size * 2, self.num_classes)

    def forward(self, x_spatial, x_temporal):
        
        # Dual Streams:
        spatial_features = self.spatial_stream.features(x_spatial)
        temporal_features = self.temporal_stream.features(x_temporal)
        
        # # Apply softmax to the outputs
        # spatial_features = F.softmax(spatial_features, dim=1)
        # temporal_features = F.softmax(temporal_features, dim=1)
        
        # Flatten features
        spatial_features = spatial_features.view(spatial_features.size(0), -1)  # [320, 25088]
        temporal_features = temporal_features.view(temporal_features.size(0), -1)  # [320, 25088]
        
        # print(f"Spatial features shape: {spatial_features.shape}")
        # print(f"Temporal features shape: {temporal_features.shape}")
        
        # Fusion Layer
        combined = torch.cat((spatial_features, temporal_features), dim=1)
        x = self.fusion_layer(combined)
        
        return x
    
    