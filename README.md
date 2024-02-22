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
- [Integration with OpenCV for Live Emotion Recognition](#integration-with-opencv-for-live-emotion-recognition)
- [Key Functions](#key-functions)
- [Libraries Used](#libraries-used)
- [Future Directions](#future-directions)

## Issues

I was able to incorporate my model with OpenCV to provide live prediction of emotions and it works great. Only issue that there are rapid fluctuations in emotion predictions frame by frame. To combat this, I will incorporate a simple moving average in hopes that this will smooth out everything.

[![Live Emotion Recognition](https://github.com/anr002/Emotion-Recognition-Study/assets/59952372/ee8252d6-ee56-439d-91b4-ebd05ee182f1)](https://www.youtube.com/watch?v=nt9DN-lKzQ0)

## Current Updates

I am currently working to combine the power of OpenCV and my model in order to predict emotions of actors in movie clips for live emotion recognition.

![image](https://github.com/anr002/Emotion-Recognition-Study/assets/59952372/2d08fe43-fc90-49cc-a254-bb02d2eacff2)

## Overview

This project aims to develop a highly accurate and efficient model for emotion recognition from facial expressions, leveraging deep learning and particularly Convolutional Neural Networks (CNNs). The goal is to understand the nuances of emotional expression and improve the accuracy of emotion detection in real-world scenarios.

## Dataset

The dataset used in this study is from FER-2013, which consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.

## Challenges

Initial models suffered from overfitting and low test accuracy, prompting iterative improvements focusing on model architecture, data preprocessing, and training strategies.

## Model Architecture

### Custom CNN

The backbone of the emotion recognition system is a custom CNN architecture designed to capture the subtle features of facial expressions. This model includes several convolutional layers, batch normalization, and dropout layers to enhance learning and prevent overfitting.

## Data Preprocessing

### Normalization and HDF5 Storage

A critical step in the preprocessing pipeline is image normalization. By scaling pixel values to a standard range, the model trains more efficiently and avoids biases towards particular intensity ranges.

## Hyperparameter Tuning with Optuna

To optimize the model's performance, Optuna has been used for hyperparameter tuning. This process has allowed for systematic exploration of a wide range of configurations, significantly improving model accuracy and stability.

## Model Evaluation

## Integration with OpenCV for Live Emotion Recognition

To bring our emotion recognition model into the real world, we've integrated it with OpenCV, enabling live emotion prediction from video streams. This integration faced challenges, notably rapid fluctuations in emotion predictions frame by frame. To address this, we implemented a simple moving average technique, smoothing out the predictions over time and providing a more stable and accurate representation of the subject's emotional state.

Additionally, we refined our OpenCV face detection process to reduce false positives, such as fleeting detections on inanimate objects. By adjusting the `detectMultiScale` parameters and implementing a form of temporal smoothing, we've significantly improved the reliability of face detection.

## Key Functions

- `load_model`: Loads the trained emotion recognition model.
- `download_youtube_video`: Downloads videos from YouTube for processing.
- `preprocess_image`: Prepares images for emotion prediction.
- `predict_emotion`: Predicts the emotion of a given image using the model.
- `process_video_stream`: Processes video streams for live emotion recognition, integrating face detection, emotion prediction, and display functionalities.

## Libraries Used

- `torch` and `torchvision` for model implementation and image transformations.
- `cv2` (OpenCV) for video processing and face detection.
- `PIL` (Pillow) for image file operations.
- `imageio` for creating GIFs from video frames.
- `numpy` for numerical operations and data manipulation.
- `optuna` for hyperparameter optimization to enhance model performance.

## Future Directions

Moving forward, I plan to explore more advanced techniques for both face detection and emotion recognition. This includes experimenting with different model architectures, such as deeper neural networks or those incorporating attention mechanisms, to improve accuracy. Additionally, I aim to expand the dataset to include more varied facial expressions and environments to further challenge and refine the model. Implementing real-time emotion recognition in more complex video scenes and integrating with other applications for interactive experiences are also on the horizon.

