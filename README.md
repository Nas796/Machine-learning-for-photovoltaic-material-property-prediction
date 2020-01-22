# Machine learning for photovoltaic material properties predictions  

## Introduction

This repository contains a regression model based on the two-layer feedforward artificial neural network for predicting (PCE). This work demonstrates the possibility to train a neural network using descriptors for predicting photovoltaic properties. The further improvement of the performance can be achieved using different neural network regularization methods such as the bayesian regularization (see BRANN architecture).

## Datasets

## Model

The model consists of two fully-connected layers with RELU activations and a regression head. As a loss function we use the mean squared error. The model has been trained using Adam optimizer.

<img src="https://user-images.githubusercontent.com/4588093/72859687-d3ca9580-3d18-11ea-8f28-ff0e89d2940f.png" width="200">

The model has been develpoed using Keras API implemented in Tensorflow.

## Results of evaluation

The squared correlation coeficient for the test set equals 0.73. Below is more information on the model performance on the train and test sets.

<img src="https://user-images.githubusercontent.com/4588093/72859688-d3ca9580-3d18-11ea-82eb-b5919efbe629.png" width="300">
<img src="https://user-images.githubusercontent.com/4588093/72859689-d3ca9580-3d18-11ea-842b-394be8cf4991.png" width="300">
<img src="https://user-images.githubusercontent.com/4588093/72859690-d4632c00-3d18-11ea-8d4c-99465d844cc6.png" width="300">
<img src="https://user-images.githubusercontent.com/4588093/72859692-d4632c00-3d18-11ea-8490-a994d2a68315.png" width="300">

## Authors

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details



