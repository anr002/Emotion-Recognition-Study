import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
import os
from improvedModel import ImprovedCustomCNN, load_data_from_hdf5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_save_model():
    # Best hyperparameters from Optuna
    lr = 0.0004848875518922412
    batch_size = 64
    weight_decay = 1.3352215817710577e-09
    activation = 'ReLU'
    step_size = 1
    gamma = 0.9251681828923234
    optimizer_name = 'Adam'
    num_classes = 7
    epochs = 54  # You might adjust this based on your previous findings or keep it dynamic

    # Load the data
    hdf5_path = 'emotion_data.hdf5'
    (train_images, train_labels), (test_images, test_labels) = load_data_from_hdf5(hdf5_path)

    # Convert to PyTorch tensors and create DataLoader
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = ImprovedCustomCNN(num_classes=num_classes, activation=activation).to(device)

    # Define optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model_save_dir = 'C://Users//andre//Documents'
    model_save_path = os.path.join(model_save_dir, 'emotion_recognition_model.pth')

    # Ensure the directory exists
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully at {model_save_path}!")

if __name__ == "__main__":
    train_and_save_model()