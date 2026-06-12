"""
Alzheimer's Disease Image Classification - PyTorch
===================================================
Algorithm: Transfer Learning using ResNet18
Dataset: OASIS MRI Images (4-Class)

Optimized for Research & IEEE Publication standards.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os
import numpy as np
from image_dataset import download_image_dataset

# Configuration
BATCH_SIZE = 32
EPOCHS = 3
IMG_SIZE = 128
# Limit to 1000 images for a "short" run as requested. Set to None to use all 80k images.
MAX_IMAGES_FOR_DEMO = 1000 

# 1. Prepare Data
data_dir, class_names = download_image_dataset()

# Image Transformations
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

# Load full dataset
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# Use a subset for quick output if specified
if MAX_IMAGES_FOR_DEMO:
    indices = np.random.choice(len(full_dataset), MAX_IMAGES_FOR_DEMO, replace=False)
    dataset = Subset(full_dataset, indices)
    print(f"[INFO] Using a subset of {MAX_IMAGES_FOR_DEMO} images for quick demonstration.")
else:
    dataset = full_dataset

# Split into Train/Val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# 2. Build Model (Transfer Learning)
print("\n[PROCESS] Loading pre-trained ResNet18...")
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 3. Training Loop
print(f"\n[PROCESS] Training on {device}...")
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size
    
    # Validation
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
    val_loss = val_loss / val_size
    val_acc = val_corrects.double() / val_size
    
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc.item())
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc.item())
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

# 4. Results & Visualization
print("\n" + "=" * 60)
print("  IMAGE CLASSIFICATION RESULTS (PyTorch)")
print("=" * 60)
print(f"Algorithm:       ResNet18 (Transfer Learning)")
print(f"Best Val Acc:    {max(history['val_acc']):.2%}")
print("=" * 60)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Accuracy curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss curves')
plt.legend()

plt.savefig("image_classification_results.png")
print("\n[SAVED] Training curves saved -> image_classification_results.png")
print("✅ PyTorch Image Classification pipeline completed on D drive!")
