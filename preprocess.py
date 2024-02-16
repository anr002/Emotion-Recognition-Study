import os
import numpy as np
from PIL import Image
import h5py

# Ensure emotion_to_label is accessible or passed to functions that need it
emotion_to_label = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

def save_to_hdf5(images, labels, filenames, dataset_name, hdf5_file):
    group = hdf5_file.create_group(dataset_name)
    group.create_dataset("images", data=images)
    
    # Convert labels from emotion strings to integers
    label_ids = np.array([emotion_to_label[label] for label in labels], dtype=np.int8)
    group.create_dataset("labels", data=label_ids)

    # Store filenames. No need to encode as ASCII, h5py handles variable-length strings well
    dt = h5py.special_dtype(vlen=str)  # Define dtype for variable-length strings
    group.create_dataset("filenames", data=np.array(filenames, dtype=dt))

def load_images_from_folder(folder, dataset_name, hdf5_file):
    images = []
    labels = []
    filenames = []
    for emotion in os.listdir(folder):
        emotion_folder = os.path.join(folder, emotion)
        if not os.path.isdir(emotion_folder):
            continue
        for filename in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('L')  # Convert to grayscale
                    if img.size != (48, 48):
                        img = img.resize((48, 48))
                    img_array = np.array(img) / 255.0
                    img_array = img_array[:, :, np.newaxis]  # Add channel dimension
                    images.append(img_array)
                    labels.append(emotion)  # This will be converted to int later
                    filenames.append(filename)
            except IOError:
                print(f'Error opening or reading image file: {img_path}')
                continue
    # Save the data to HDF5
    save_to_hdf5(np.array(images), np.array(labels), filenames, dataset_name, hdf5_file)

# Define the path to your dataset
base_path = 'C:/Users/andre/Downloads/archive'

# Create a new HDF5 file
hdf5_path = 'emotion_data.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')

try:
    # Load the training and test data
    load_images_from_folder(os.path.join(base_path, 'train'), 'train', hdf5_file)
    load_images_from_folder(os.path.join(base_path, 'test'), 'test', hdf5_file)

    # Store label mapping as an attribute in a more accessible format
    hdf5_file.attrs['label_mapping'] = str(emotion_to_label)
finally:
    hdf5_file.close()