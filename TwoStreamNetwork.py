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
        
        #TODO: Load pretrained model
                
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
        
        #TODO: Load pretrained model method

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
        self.fusion_layer = nn.Linear(2*self.num_classes, self.num_classes)

    def forward(self, x_spatial, x_temporal):
        
        # Dual Streams:
        spatial_features = self.spatial_stream.features(x_spatial)
        temporal_features = self.temporal_stream.features(x_temporal)
        
        # # Apply softmax to the outputs
        # spatial_features = F.softmax(spatial_features, dim=1)
        # temporal_features = F.softmax(temporal_features, dim=1)
        
        # Fusion Layer
        x = torch.cat((spatial_features, temporal_features), dim=1)
        
        # Classification Layer
        x = self.classifier(x)
        
        return x
    
    
    
# root_dir = 'ufc10'
    
# transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
# frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
# framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
# framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


# frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
# framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
# framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)
# # Initialize the model
# model = EarlyFusion(num_classes=10)

# # Define a loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ############################# TESTING #############################
# if __name__ == '__main__':
#     # Hyperparameters
#     hyperparameters = {
#         'num_classes': 10,
#     }

#     # Create models with different fusion types
#     early_fusion_model = EarlyFusionModel(hyperparameters, load_pretrained=True, use_lstm=True)

#     # Input (batch_size, channels, frames, height, width)
#     x = torch.randn(8, 3, 10, 64, 64)

#     # Test MLP fusion
#     early_out = early_fusion_model(x)
#     print("Early Output Shape:", early_out.shape)
