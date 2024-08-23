# Plant Disease Classification Model

This repository contains the code for building and training a Convolutional Neural Network (CNN) model to classify different diseases in plants using images. The dataset used in this project is sourced from the PlantVillage dataset on Kaggle.

## Dataset

The dataset used in this project is the [PlantVillage dataset](https://www.kaggle.com/arjuntejaswi/plant-village), which contains images of healthy and diseased leaves. The images are categorized into three classes:
1. **Healthy**
2. **Early Blight**
3. **Late Blight**

## Dependencies

The project requires the following dependencies:

- Python 3.x
- TensorFlow
- Matplotlib
- Numpy

To install the required packages, run:

```bash
pip install tensorflow matplotlib numpy
```

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. It consists of several convolutional layers with ReLU activation, followed by max-pooling layers. The final layers are fully connected, ending with a softmax activation function to output the class probabilities.

### Layers

- **Input Layer**: Resizes images to 256x256 and rescales pixel values to [0, 1].
- **Convolutional Layers**: Six convolutional layers with varying numbers of filters (32, 64).
- **Max-Pooling Layers**: Added after each convolutional layer to reduce the spatial dimensions.
- **Flatten Layer**: Converts the 2D matrix into a 1D vector.
- **Dense Layers**: Two dense layers, the first with 64 neurons and ReLU activation, and the final output layer with softmax activation for classification.

## Data Augmentation

To improve the model's performance, data augmentation techniques such as random flipping and random rotation were applied to the training dataset.

## Training

The model was trained for 50 epochs with a batch size of 32. The dataset was split into training (80%), validation (10%), and testing (10%) sets. The `adam` optimizer and `SparseCategoricalCrossentropy` loss function were used during training.

## Evaluation

The model achieved the following scores on the test dataset:

- **Test Loss**: 0.0063
- **Test Accuracy**: 1.0000

## Results

The training and validation accuracy and loss curves are plotted to visualize the model’s performance over the epochs. The final model can accurately classify diseases from the test set images.

## Inference

The notebook includes functions for running predictions on new images. The `predict` function takes a model and an image as input and returns the predicted class along with the confidence level.

## Acknowledgements

- Dataset credits: [PlantVillage](https://www.kaggle.com/arjuntejaswi/plant-village)
- Framework: TensorFlow/Keras
