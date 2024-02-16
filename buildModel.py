import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),  # Adjust the input features to match your data
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def load_data_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        train_images = np.array(hdf5_file['train/images']).astype('float32')
        train_labels = np.array(hdf5_file['train/labels']).astype('long')
        test_images = np.array(hdf5_file['test/images']).astype('float32')
        test_labels = np.array(hdf5_file['test/labels']).astype('long')
        
    # Ensure the images are in the correct shape for PyTorch (N, C, H, W)
    train_images = train_images.reshape(-1, 1, 48, 48)  # Reshape train images
    test_images = test_images.reshape(-1, 1, 48, 48)  # Reshape test images
    
    # Convert to PyTorch tensors
    train_images = torch.tensor(train_images)
    train_labels = torch.tensor(train_labels).long()  # Convert labels to Long type
    test_images = torch.tensor(test_images)
    test_labels = torch.tensor(test_labels).long()  # Convert labels to Long type
    
    return (train_images, train_labels), (test_images, test_labels)

# Define the input shape and the number of classes
num_classes = 7

# Initialize the model and move it to the GPU if available
model = CustomCNN(num_classes).to(device)

# Load the data
hdf5_path = 'emotion_data.hdf5'
(train_images, train_labels), (test_images, test_labels) = load_data_from_hdf5(hdf5_path)

# Create DataLoader
batch_size = 64
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Move data to the same device as the model
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        # Move data to the same device as the model
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')