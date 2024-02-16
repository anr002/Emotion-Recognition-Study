import numpy as np
import h5py
import matplotlib.pyplot as plt
import random

# Path to your HDF5 file
hdf5_path = 'emotion_data.hdf5'

# Emotion labels mapping (inverse of emotion_to_label)
label_to_emotion = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

def display_random_image(dataset_name):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        # Assuming dataset_name is either 'train' or 'test'
        group = hdf5_file[dataset_name]
        
        # Get the number of images
        num_images = group['images'].shape[0]
        
        # Select a random index
        rand_index = random.randint(0, num_images - 1)
        
        # Read the image and label
        image = group['images'][rand_index]
        label_id = group['labels'][rand_index][()]
        
        # Convert the label ID to emotion name
        emotion = label_to_emotion.get(label_id, "Unknown")
        
        # Display the image and label
        plt.imshow(image.squeeze(), cmap='gray')  # Use squeeze() to remove single-dimensional entries from the shape
        plt.title(f'Emotion: {emotion}')
        plt.axis('off')  # Hide the axis
        plt.show()

# Example usage
display_random_image('train')  # or 'test'