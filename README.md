# Data Mining Project: <br>
# Emotion Detection in Real-Time Video (With KDD analysis)

## Overview
This project focuses on detecting emotions in real-time video streams using deep learning techniques. The model is trained on facial expression data to predict emotions in live video feeds.

## Knowing the Data
- The dataset comprises images of facial expressions categorized into seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise.
- Data is split into two sections: train and validation, each containing subfolders for different emotions.
- The objective is to classify facial expressions into predefined emotion categories.<br>
#### **KDD**
- **Objective**: Gain insights into the dataset and understand its structure, features, and distribution.
- **Actions**:
  - Explore the dataset to understand the distribution of facial expressions.
  - Analyze the characteristics of images in the dataset, such as resolution and quality.
  - Identify any missing or erroneous data that may require preprocessing.

## Data Preprocessing
- Data preprocessing involves several steps to prepare the dataset for model training:
  1. **Loading Images**: Images are loaded using the `load_img` function from the Keras library.
  2. **img_to_array**: This function converts a PIL image object into a NumPy array. It is used to convert the loaded images into arrays for further processing.
  3. **ImageDataGenerator**: This class generates batches of tensor image data with real-time data augmentation. It is used to perform data augmentation on the image data during training.
  4. **Data Splitting**: The dataset is split into training and validation sets for model evaluation.<br>
#### **KDD**
- **Objective**: Preprocess the dataset to ensure it is suitable for model training.
- **Actions**:
  - Load the image data and convert it into a format compatible with the chosen deep learning framework.
  - Perform data augmentation techniques to increase the diversity of the training data and improve the model's generalization.
  - Split the dataset into training and validation sets to evaluate the model's performance.
  
## Dataset Details
- The dataset is organized into two main folders: train and validation.
- Each folder contains subfolders corresponding to different emotion categories.
- Images in each subfolder represent facial expressions of the respective emotion category.

## Model Building
- The model architecture consists of a convolutional neural network (CNN) designed to classify facial expressions.
- Various CNN layers, including convolutional, pooling, and fully connected layers, are utilized to learn and extract features from facial images.
- The model is trained using the Adam optimizer with a learning rate of 0.001 over 48 epochs.<br>
#### **KDD**
- **Objective**: Developed a deep learning model to classify facial expressions in real-time video streams.
- **Actions**:
  - Designed a convolutional neural network (CNN) architecture suitable for image classification tasks.
  - Experiment with different CNN architectures, activation functions, and optimization algorithms to optimize model performance.
  - Train the model using the prepared dataset and evaluate its performance on the validation set.

## Function Explanation:
1. **load_img**: This function loads an image file and returns it as a PIL (Python Imaging Library) image object. It is used to load images from the dataset.
2. **img_to_array**: This function converts a PIL image object into a NumPy array. It is used to convert the loaded images into arrays for further processing.
3. **ImageDataGenerator**: This class generates batches of tensor image data with real-time data augmentation. It is used to perform data augmentation on the image data during training.
4. **Sequential**: This class allows you to build a sequential model layer-by-layer. It is used to create a sequential model for the CNN.
5. **Conv2D**: This class creates a convolutional layer for 2D spatial convolution. It applies a specified number of filters to the input data.
6. **BatchNormalization**: This layer normalizes the activations of the previous layer at each batch. It helps in stabilizing and accelerating the training process.
7. **Activation**: This layer applies an activation function to the output of the previous layer. Common activation functions include 'relu' (Rectified Linear Unit) and 'softmax'.
8. **MaxPooling2D**: This layer performs max pooling operation for spatial data. It reduces the spatial dimensions of the input volume.
9. **Dropout**: This layer applies dropout regularization to the input. It randomly sets a fraction of input units to zero during training to prevent overfitting.
10. **Dense**: This layer implements the operation: output = activation(dot(input, kernel) + bias). It is the standard fully connected layer.
11. **Model**: This class groups layers into an object with training and inference features. It is used to define the model architecture and compile it for training.
12. **Adam**: This optimizer is an extension to stochastic gradient descent. It computes adaptive learning rates for each parameter.
13. **ModelCheckpoint**: This callback saves the model after every epoch if the validation accuracy improves.
14. **EarlyStopping**: This callback stops training when a monitored metric has stopped improving.
15. **ReduceLROnPlateau**: This callback reduces the learning rate when a metric has stopped improving.

## Working of the Code:
1. The code starts by importing necessary libraries and setting parameters such as image size, folder path, and target emotions.
2. It loads images from the specified folder path using `load_img` and `img_to_array`.
3. The dataset is split into training and validation sets using `ImageDataGenerator` and `flow_from_directory`.
4. A CNN model architecture is defined using `Sequential` and various layers such as `Conv2D`, `BatchNormalization`, `Activation`, etc.
5. The model is compiled with the Adam optimizer and categorical cross-entropy loss function.
6. Callbacks such as `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau` are defined to monitor the training process.
7. The model is trained using the `fit_generator` function, which iterates over the training set for a specified number of epochs.
8. Training and validation loss/accuracy curves are plotted using `matplotlib`.

This code serves as a foundation for building an emotion detection model and can be further extended and optimized for improved performance.


## Performance Metrics
- Training and validation loss/accuracy curves are plotted to evaluate the model's performance.
- The model achieves a test accuracy of 65%, significantly outperforming random guessing.

## Predictive Modeling
- A predictive model is built to classify emotions in real-time video streams.
- Key variables such as facial features, expressions, and historical data are considered in predicting emotions.
- Various classifiers including Logistic Regression, Support Vector Machines, Decision Tree, etc., are tested to determine the best-performing model.

## Results
- The final model achieves a test accuracy of 65%, demonstrating its effectiveness in real-time emotion detection.
- The model's performance is compared to a baseline predictor, showing a significant improvement in accuracy.<br>
#### **KDD**
- **Objective**: Assess the performance of the trained model and identify areas for improvement.
- **Actions**:
  - Evaluate the model's accuracy, precision, recall, and F1-score on the validation set.
  - Use techniques such as confusion matrices and ROC curves to analyze the model's performance across different emotion categories.
  - Identify any misclassifications or patterns in the model's predictions and refine the model accordingly.
    
## Future Work
- Implement real-time video processing optimizations for faster inference.
- Extend the model to detect and classify complex emotional states in diverse scenarios.

## Procedure to run:
1.Libraries Required:<br>
    Keras <br>
    tenserflow<br>
    pandas<br>
    scikit-learn<br>
    numpy<br>
    matplotlib<br>
2.Open main.py file.<br>

Data Set Link - https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset
