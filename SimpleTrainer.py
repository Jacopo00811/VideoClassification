import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import FrameVideoDataset
from model3d import TheConvolver3D
import torch.optim as optim

root_dir = "/dtu/blackhole/0d/203501/Data/IDLCV/ufc10"

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

num_epochs = 30  # You can adjust the number of epochs

wandb.init(
    project="IDLCV",
    config={
        "learning_rate": optimizer.param_groups[0]['lr'],
        "architecture": str(model.__class__.__name__),
        "dataset": str(train_loader.name) if hasattr(train_loader, 'name') else "No dataset name",
        "epochs": epochs,
        "batch_size": train_loader.batch_size if hasattr(train_loader, 'batch_size') else "No batchsize",
        "optimizer": optimizer.__class__.__name__,
        "loss_fn": model.loss_fn.__class__.__name__ if hasattr(model, "loss_fn") else "No loss function",
    }
)
best_test_accuracy = -float('inf')
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
        
        # # Print statistics every 10 batches
        # if (batch_idx + 1) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
        #           f'Loss: {running_loss / 10:.4f}, '
        #           f'Accuracy: {100 * correct_predictions / total_predictions:.2f}%')
            # running_loss = 0.0  # Reset running loss
    # Calculate average loss and accuracy for the epoch
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_predictions / total_predictions

    # Evaluation phase
    model.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    # Calculate average test loss and accuracy for the epoch
    avg_test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    print(f"Epoch {epoch + 1}/{epochs} | Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")
    
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), f"{model.__class__.__name__}_best.pth")
        print(f"Model saved with Test Accuracy: {test_accuracy:.4f}")
    # Log metrics to wandb
    wandb.log({
        "train_loss": avg_train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

wandb.finish()