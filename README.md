# Digit-Recognition-System-using-TensorFlow-and-CNN

This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

## Dataset
- **MNIST**: 70,000 grayscale images of handwritten digits (0–9), each 28x28 pixels.
- Split into 60,000 training and 10,000 testing images.

## Model Architecture
- **Conv2D** (32 filters, 3x3, ReLU)
- **MaxPooling2D** (2x2)
- **Flatten**
- **Dense** (128 neurons, ReLU)
- **Dense** (10 neurons, Softmax)

## Steps
1. Load and preprocess the MNIST dataset.
2. Normalize pixel values to [0, 1].
3. Build and compile the CNN model.
4. Train for 5 epochs with validation.
5. Evaluate on test data.

## Requirements
- Python 3.x
- TensorFlow
- NumPy

## Usage
```bash
pip install tensorflow numpy
python mnist_cnn.py
