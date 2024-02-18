import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
import torch.nn.functional as F

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImprovedCustomCNN(nn.Module):
    def __init__(self, num_classes, activation='ReLU'):
        super(ImprovedCustomCNN, self).__init__()
        # Store the activation function
        self.activation = self._get_activation(activation)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding='same'),
            self.activation(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            self.activation(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            self.activation(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            self.activation(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            self.activation(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            self.activation(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _get_activation(self, name):
        """Returns the activation function based on the name."""
        activations = {
            'ReLU': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'ELU': nn.ELU
        }
        return activations.get(name, nn.ReLU)  # Default to ReLU if name not found

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
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

# Example usage
if __name__ == "__main__":
    num_classes = 7
    model = ImprovedCustomCNN(num_classes).to(device)
    hdf5_path = 'emotion_data.hdf5'
    (train_images, train_labels), (test_images, test_labels) = load_data_from_hdf5(hdf5_path)
    batch_size = 64
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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