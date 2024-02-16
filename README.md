# Emotion Recognition Study

## Table of Contents
- [Current Updates](#current-updates)
- [Overview](#overview)
- [Dataset](#dataset)
- [Challenges](#challenges)
- [Model Architecture](#model-architecture)
  - [Custom CNN](#custom-cnn)
- [Data Preprocessing](#data-preprocessing)
  - [Normalization and HDF5 Storage](#normalization-and-hdf5-storage)
- [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)
- [Model Evaluation](#model-evaluation)
- [Future Directions](#future-directions)

# Emotion Recognition Study

## Current Updates

Recent efforts have focused on enhancing the model's performance by expanding the range of tuned hyperparameters. This approach has successfully increased the model's accuracy to 62.4%. The key adjustments involved fine-tuning learning rate, batch size, weight decay, activation functions, and learning rate scheduler parameters, pinpointing an optimal learning rate around 0.0007 and a batch size of 32, among other specifics.

The next step involves delving into the CNN architecture. The plan is to experiment with layer configurations and activation functions, and to consider integrating techniques like data augmentation and additional regularization methods. The aim is to further refine the model's accuracy and ensure its effectiveness in real-world scenarios.


## Overview

This project aims to develop a highly accurate and efficient model for emotion recognition from facial expressions, leveraging deep learning and particularly Convolutional Neural Networks (CNNs). The goal is to understand the nuances of emotional expression and improve the accuracy of emotion detection in real-world scenarios.

## Dataset

The dataset used in this study is from FER-2013, which consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.

The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples, and the public test set consists of 3,589 examples.

## Challenges

Initial models suffered from overfitting and low test accuracy, prompting iterative improvements focusing on model architecture, data preprocessing, and training strategies.

## Model Architecture

### Custom CNN

The backbone of the emotion recognition system is a custom CNN architecture designed to capture the subtle features of facial expressions. This model includes several convolutional layers, batch normalization, and dropout layers to enhance learning and prevent overfitting.

## Data Preprocessing

### Normalization and HDF5 Storage

A critical step in the preprocessing pipeline is image normalization. By scaling pixel values to a standard range, the model trains more efficiently and avoids biases towards particular intensity ranges. To manage the dataset efficiently, especially considering its size and the need for high-throughput access during training, the data is stored in an HDF5 file. This choice significantly improves data loading times and overall training efficiency. The `preprocess.py` script handles the normalization and storage, ensuring that each image is correctly processed and readily accessible for training.

## Hyperparameter Tuning with Optuna

To optimize the model's performance, Optuna has been employed for hyperparameter tuning. This process has allowed for systematic exploration of a wide range of configurations, significantly improving model accuracy and stability.

## Model Evaluation


## Future Directions

