import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import FrameImageDataset
from aggregation_model import Pretrained
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import os

# Initialize Weights & Biases
wandb.init(
    project="IDLCV",
    config={
        "learning_rate": 0.001,  # Initial learning rate
        "architecture": "TheAggregator",
        "dataset": "ufc10",
        "epochs": 50,
        "batch_size": 8,
        "optimizer": "Adam",
        "loss_fn": "CrossEntropyLoss",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_params": {
            "mode": "max",        # 'max' for accuracy, 'min' for loss
            "factor": 0.1,        # Reduce LR by a factor of 0.1
            "patience": 5,        # Wait for 5 epochs with no improvement
            "threshold": 0.01,    # Minimum change to qualify as an improvement
            "verbose": True,
            "min_lr": 1e-6        # Minimum learning rate
        }
    },
    name="Aggregation",
)

# Define the root directory
cwd = os.getcwd()
root_dir = os.path.join(cwd, "ufc10")
# root_dir = "/dtu/blackhole/03/148387/ufc10"

transform = transforms.Compose([
    transforms.Resize((250, 250)),  # Ensure images are at least slightly larger than crop size
    transforms.RandomResizedCrop(224,scale=(0.8, 1.0), ratio=(0.75, 1.33)), # Better size for ResNet101
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
val_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
test_dataset = FrameImageDataset(root_dir=root_dir, split='test', transform=transform)
# framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
# framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


train_loader = DataLoader(train_dataset,  batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset,  batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=4)

model = Pretrained("VGG16",pretrained=True, freeze_backbone=True)

# Move model to the specified device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4,weight_decay = 1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

# Create save folder
os.makedirs(os.path.dirname("saved_models"), exist_ok=True)

epochs = 35

best_test_accuracy = -float('inf')  # Track the best test loss to save the model

for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)

        # Forward pass
        outputs = model(inputs)
        if outputs.dim() > 1 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)  # Squeeze only the channel dimension
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Calculate accuracy for this batch
        # if not hasattr(model, "activation") or model.activation is None:
        #     raise ValueError("The model does not have an activation function defined!")
        predicted = torch.argmax(outputs, dim=1)

        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    # Calculate average loss and accuracy for the epoch
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {avg_train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%")

    # Evaluation phase
    model.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)  # Squeeze only the channel dimension

            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)

            # Accumulate test loss
            running_loss += loss.item()

            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    # Calculate average test loss and accuracy for the epoch
    avg_test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{epochs} | Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

    # Update the learning rate using the scheduler
    if scheduler:
        scheduler.step(avg_test_loss)  # Adjust based on test loss (for ReduceLROnPlateau)

    # Save the best model
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy': best_test_accuracy,
        }, f"saved_models/{model.__class__.__name__}_best.pth")
        print(f"Model saved with Test Accuracy: {test_accuracy:.2f}%")

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
        "learning_rate": current_lr
    })

wandb.finish()