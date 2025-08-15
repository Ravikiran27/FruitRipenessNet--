import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

# Paths to your data folders
train_dir = 's:/project/data/Train'
test_dir = 's:/project/data/Test'

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and loaders
datasets_train = datasets.ImageFolder(train_dir, transform=transform)
datasets_test = datasets.ImageFolder(test_dir, transform=transform)
train_loader = DataLoader(datasets_train, batch_size=32, shuffle=True)
test_loader = DataLoader(datasets_test, batch_size=32, shuffle=False)

# Simple CNN Model
class FruitRipenessCNN(nn.Module):
    def __init__(self, num_classes):
        super(FruitRipenessCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Instantiate model
num_classes = len(datasets_train.classes)
model = FruitRipenessCNN(num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

print('Training complete.')

# Save model
torch.save(model.state_dict(), 'fruit_ripeness_cnn.pth')
