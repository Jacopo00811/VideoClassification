import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import FrameVideoDataset
from model3d import TheConvolver3D
import torch.optim as optim

root_dir = 'ufc10'

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

# Create the training and validation datasets
train_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
val_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

import torch.optim as optim

# Initialize the model
model = TheConvolver3D()

# Move the model to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 5  # You can adjust the number of epochs

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Compute accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        # Print statistics every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {running_loss / 10:.4f}, '
                  f'Accuracy: {100 * correct_predictions / total_predictions:.2f}%')
            running_loss = 0.0  # Reset running loss
