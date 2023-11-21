
# Aerial Semantic Segmentation Using U-Net Architecture

This repository contains the code and documentation for a project on Aerial Semantic Segmentation using the U-Net architecture.

## Motivation

Aerial semantic segmentation is a cutting-edge technology with applications in environmental monitoring, urban planning, agriculture, disaster management, and more. It involves partitioning high-resolution aerial images into meaningful segments, each labeled with the corresponding object or land cover class. This project aims to explore and implement the U-Net architecture for accurate and detailed aerial semantic segmentation.

## Introduction

Aerial Semantic Segmentation classifies image pixels into semantic groups or classes, specifically in the context of aerial images. The process involves several steps:

1. **Data Acquisition:** The dataset, consisting of 800 images with corresponding labeled images, was sourced from Kaggle. Each image has a resolution of 6000 x 4000 pixels, providing a diverse range of scenarios for robust model training.

2. **Data Pre-processing:** The dataset underwent resizing, cropping, data augmentation, and normalization for efficient model training. Class imbalance was addressed, and the dataset was split into training, validation, and testing sets.

3. **Semantic Labelling (Training):** Deep learning models, particularly Convolutional Neural Networks (CNNs), were used for aerial semantic segmentation. The model learned to recognize patterns and features corresponding to semantic classes during training.

4. **Inference (Testing):** The trained model performed inference on new, unlabeled aerial images, assigning a class label to each pixel based on its learned understanding.

## U-Net Architecture

The U-Net architecture consists of an encoder (contracting path), bottleneck, and decoder (expansive path). Skip connections are used to fuse features from different layers, improving segmentation accuracy, especially along object boundaries.

### Advantages of U-Net for Aerial Semantic Segmentation

- Able to learn accurate and detailed segmentation masks even from small datasets.
- Robust to noise and artifacts in aerial images.
- Capable of segmenting a wide variety of objects in aerial images.

### Disadvantages of U-Net for Aerial Semantic Segmentation

- Can be computationally expensive to train.
- Sensitive to hyperparameters.
- Interpretability of predictions can be challenging.

## Training Procedure

1. Load the training dataset.
2. Initialize the U-Net model.
3. Set the loss function and optimizer.
4. For each epoch, iterate over the dataset, performing forward pass, loss calculation, and backward pass.
5. Evaluate the model on the validation dataset.

## Code

The code for implementing and training the U-Net model is available in the provided LaTeX document. Additionally, the code was executed in Google Colab for GPU acceleration.

## Results

- For images of size 384 * 512:
  - Final Training Accuracy: 94.65%
  - Testing Loss: 0.574, Testing Accuracy: 86.41%

<!-- Include any additional results or findings here -->

## References

1. [U-Net Paper](https://arxiv.org/pdf/1505.04597v1.pdf)
2. [Aerial Semantic Segmentation Drone Dataset](http://dronedataset.icg.tugraz.at)
3. [U-Net Model Implementation on Kaggle](https://www.kaggle.com/code/yesa911/aerial-semantic-segmentation-96-acc/notebook)
```

Feel free to customize the content based on additional information or specific details you want to include in your README file.
