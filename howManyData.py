import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np

# Assuming the load_data_from_hdf5 function is defined in your script
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
# Load the data
hdf5_path = 'emotion_data.hdf5'
(train_images, train_labels), (test_images, test_labels) = load_data_from_hdf5(hdf5_path)

# Calculate the number of data points
num_train_data_points = len(train_images)
num_test_data_points = len(test_images)
total_data_points = num_train_data_points + num_test_data_points

print(f"Number of training data points: {num_train_data_points}")
print(f"Number of testing data points: {num_test_data_points}")
print(f"Total number of data points: {total_data_points}")