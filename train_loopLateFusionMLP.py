import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import FrameVideoDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from late_fusion import LateFusion
import wandb
import os
from tqdm import tqdm  # Import tqdm

# Initialize Weights & Biases
wandb.init(
    project="IDLCV",
    config={
        "learning_rate": 0.0001,  # Initial learning rate
        "architecture": "LateFusion",
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
        },
    },
    name="late_fusionMLP",
)

hyperparameters = {
    'num_classes': 10,
}
cwd = os.getcwd()

# Define the root directory
root_dir = os.path.join(cwd, "ufc10")
# root_dir = "../ufc10"

# Define transformations with data augmentation (optional)
transform = T.Compose([
    T.Resize((64, 64)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Create the training and validation datasets
train_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
val_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize the model
model = LateFusion(hyperparameters=hyperparameters, load_pretrained=True, fusion_type='mlp')

# Move the model to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Define the learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',          # Change to 'min' if monitoring loss
    factor=0.1,
    patience=10,
    threshold=0.01,
    verbose=True,
    min_lr=1e-6
)

# Number of epochs
epochs = 100  # You can adjust the number of epochs

# To keep track of the best validation accuracy
best_test_accuracy = -float('inf')

# Ensure a directory exists to save models
os.makedirs('saved_models', exist_ok=True)

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # Add tqdm to the training loop
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
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
    
    # Calculate average loss and accuracy for the epoch
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_predictions / total_predictions

    # Evaluation phase
    model.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0

    # Add tqdm to the validation loop
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}")):
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
    avg_test_loss = running_loss / len(val_loader)
    test_accuracy = 100 * correct_test / total_test
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{epochs} | Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}% | Learning Rate: {current_lr}")
    
    # Step the scheduler based on validation accuracy
    scheduler.step(test_accuracy)
    
    # Check if this is the best model so far
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        # Save the best model
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

# Could also try this optimizer and scheduler combination
# def get_optimizer_and_scheduler(model, num_epochs):
#     optimizer = torch.optim.AdamW(model.parameters(), 
#                                  lr=1e-4, 
#                                  weight_decay=0.01)
    
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr=1e-3,
#         epochs=num_epochs,
#         steps_per_epoch=steps_per_epoch,
#         pct_start=0.1  # 10% warmup
#     )
    
#     return optimizer, scheduler