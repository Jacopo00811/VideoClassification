# Make Early Fusion Model:
# Importing Required Libraries:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from datasets import FrameImageDataset, FrameVideoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class EarlyFusionModel(nn.Module):
    def __init__(self, hyperparameters, load_pretrained=True, use_lstm=False):
        super(EarlyFusionModel, self).__init__()
        self.num_classes = hyperparameters['num_classes']
        self.use_lstm = use_lstm
        self.load(load_pretrained)
        
        # Replace the final fully connected layer with a new one that has num_classes outputs
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove the final layer
        
        if self.use_lstm:
            # Add an LSTM layer
            self.lstm = nn.LSTM(in_features, 512, batch_first=True)
            self.fc = nn.Linear(512, self.num_classes)
        else:
            # Directly add a fully connected layer
            self.fc = nn.Linear(in_features, self.num_classes)
        
    def load(self, load_pretrained):
        if load_pretrained:
            self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            self.model = models.resnet152(weights=None)
            
    def forward(self, x):
        batch_size, channels, time, height, width = x.size()
        x = x.view(batch_size * time, channels, height, width)
        
        # Pass through the base model
        x = self.model(x)
        
        # Reshape back to (batch_size, time, in_features)
        x = x.view(batch_size, time, -1)
        
        if self.use_lstm:
            # Pass through the LSTM
            x, _ = self.lstm(x)
            
            # Take the output of the last time step
            x = x[:, -1, :]
        else:
            # Average over the time dimension
            x = x.mean(dim=1)
        
        # Pass through the final fully connected layer
        x = self.fc(x)
        
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
