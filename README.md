# CIFAR-10-Image-Classification-with-CNN-and-Result-Visualization

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset.

# Overview
The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes (airplane, car, bird, cat, etc.). This project builds a CNN to classify these images, including data preprocessing, training, evaluation, and result visualization.

# Model Architecture
- Conv2D layers with ReLU activation
- MaxPooling2D for downsampling
- BatchNormalization for training stability
- Dropout for overfitting prevention
- Dense layer with softmax for classification

# Training
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- EarlyStopping is used to prevent overfitting.

# Evaluation
- Confusion matrix and accuracy are used for performance evaluation.
- Visualizations of predictions and the confusion matrix are generated.

# Features
- Data Preprocessing: The input images are normalized by scaling pixel values between 0 and 1 to speed up training and ensure numerical stability.
- One-Hot Encoding: The class labels are one-hot encoded to suit the softmax output of the network.
- Early Stopping: Early stopping is applied to halt training once validation performance stops improving, preventing overfitting.

# Result Visualization
1. **Training Progress**: 
   - Plots are generated to visualize the model’s accuracy and loss for both training and validation sets, allowing you to monitor performance across epochs.
   
2. **Confusion Matrix**: 
   - A confusion matrix is plotted to show how well the model distinguishes between the 10 different CIFAR-10 classes.
   - This helps identify which categories the model struggles with and which are well-predicted.

3. **Sample Predictions**:
   - A set of images from the test data is visualized along with their predicted labels, providing insight into how the model performs on individual samples.
