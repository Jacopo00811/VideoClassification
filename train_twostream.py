import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets_new import FrameVideoDataset, FlowVideoDataset
from TwoStreamNetwork import TwoStream
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# Configuration
config = dict(
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    num_frames=10,
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    hyperparameters={
        'num_classes': 10
    },
    optimizer_config={
        "optimizer": "Adam",
        "loss_fn": "CrossEntropyLoss",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_params": {
            "mode": "max",
            "factor": 0.1,
            "patience": 5,
            "threshold": 0.01,
            "verbose": True,
            "min_lr": 1e-6
        },
    }
)

def get_transforms():
    spatial_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])
    
    flow_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return spatial_transform, flow_transform

def get_dataloaders(root_dir, batch_size, num_frames):
    spatial_transform, flow_transform = get_transforms()
    
    # Create datasets
    spatial_dataset = FrameVideoDataset(root_dir=os.path.join(root_dir, "frames"))
    flow_dataset = FlowVideoDataset(root_dir=os.path.join(root_dir, "flows"))
    
    # Create dataloaders
    spatial_loader = DataLoader(spatial_dataset, batch_size=batch_size, shuffle=True)
    
    flow_loader = DataLoader(flow_dataset, batch_size=batch_size, shuffle=True)
    
    return spatial_loader, flow_loader

def train_epoch(model, spatial_loader, flow_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for (spatial_frames, _), (flow_frames, labels) in zip(spatial_loader, flow_loader):
        # Move to device and get dimensions
        spatial_frames = spatial_frames.to(device)  # [B, C, T, H, W]
        flow_frames = flow_frames.to(device)       # [B, C_flow, H, W]
        labels = labels.to(device)
        
        B, C, T, H, W = spatial_frames.size()
        
        # print(f"Original spatial shape: {spatial_frames.shape}")
        # print(f"Original flow shape: {flow_frames.shape}")
        
        # Handle spatial frames
        spatial_frames = spatial_frames.transpose(1, 2)  # [B, T, C, H, W]
        spatial_frames = spatial_frames.reshape(-1, C, H, W)  # [B*T, C, H, W]
        
        # Get flow dimensions 
        B_flow, C_flow, H_flow, W_flow = flow_frames.size()  # [32, 18, 64, 64]
        # print(f"Flow frame shape: {flow_frames.shape}")
        
        # Handle temporal frames - replicate across time dimension
        flow_frames = flow_frames.unsqueeze(1)  # [B, 1, C_flow, H, W]
        flow_frames = flow_frames.expand(-1, T, -1, -1, -1)  # [B, T, C_flow, H, W]
        flow_frames = flow_frames.reshape(B*T, -1, H_flow, W_flow)  # [B*T, C_flow, H, W]
        
        # Resize flow frames to match spatial resolution
        if H_flow != H or W_flow != W:
            flow_frames = F.interpolate(flow_frames, size=(H, W), 
                                      mode='bilinear', align_corners=False)
        
        # print(f"Processed spatial shape: {spatial_frames.shape}")
        # print(f"Processed flow shape: {flow_frames.shape}")
        
        optimizer.zero_grad()
        outputs = model(spatial_frames, flow_frames)  # [B*T, num_classes]
        
        # Average predictions across time dimension
        outputs = outputs.view(B, T, -1)  # [B, T, num_classes]
        outputs = outputs.mean(dim=1)     # [B, num_classes]
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return total_loss / len(spatial_loader), 100. * correct / total

def main():
    wandb.init(project="IDLCV", config=config, name="two_streamNoLeaks")
    
    # Setup
    device = config['device']
    cwd = os.getcwd()

    # Define the root directory
    root_dir = os.path.join(cwd, "ufc10")
    # root_dir = "../ufc10"
    
    # Get dataloaders
    spatial_loader, flow_loader = get_dataloaders(
        root_dir, 
        config['batch_size'],
        config['num_frames']
    )
    
    # Initialize model
    model = TwoStream(
        hyperparameters=config['hyperparameters'],
        timesteps=config['num_frames']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, **config['optimizer_config']['scheduler_params'])
    
    # Training loop
    best_acc = 0
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(
            model, spatial_loader, flow_loader, 
            criterion, optimizer, device
        )
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "epoch": epoch
        })
        
        # Save checkpoint if best accuracy
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'saved_models/TwoStream_best_model.pth')
        
        # Update learning rate
        scheduler.step(train_acc)
        
if __name__ == '__main__':
    main()