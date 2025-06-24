import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from PIL import Image
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_dir = "C:/Users/khush/Downloads/archive (1)/dataset_blood_group"

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
full_data = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = full_data.classes
image_paths = [s[0] for s in full_data.samples]

# Class weights
labels = [label for _, label in full_data]
label_counts = Counter(labels)
total_count = sum(label_counts.values())
class_weights = [total_count / label_counts[i] for i in range(len(class_names))]
weights_tensor = torch.FloatTensor(class_weights).to(device)

# Split the dataset
total_size = len(full_data)
val_size = test_size = int(0.15 * total_size)
train_size = total_size - val_size - test_size
train_data, val_data, test_data = random_split(full_data, [train_size, val_size, test_size],
                                               generator=torch.Generator().manual_seed(42))

# Save unseen test image paths
test_image_paths = [image_paths[i] for i in test_data.indices]
pd.DataFrame(test_image_paths, columns=["unseen_test_images"]).to_csv("test_images.csv", index=False)

# Data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# CNN Model
class BloodGroupCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(BloodGroupCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Training setup
model = BloodGroupCNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_val_acc = 0.0
train_loss_history = []
val_loss_history = []
val_accuracy_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    train_loss_history.append(epoch_train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = 100 * correct / total
    val_loss_history.append(epoch_val_loss)
    val_accuracy_history.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

    # Save best model
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), 'app/model/blood_group_model_best.pth')

# Final evaluation
model.load_state_dict(torch.load('app/model/blood_group_model_best.pth'))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nTest Set Performance:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label='Train Loss', marker='o')
plt.plot(val_loss_history, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
