# Machine learning for photovoltaic material properties predictions  

## Introduction

This repository contains a regression model based on the two-layer feedforward artificial neural network for predicting the power conversion efficiency (PCE). This work demonstrates the possibility to train a neural network using descriptors for predicting photovoltaic properties. The further improvement of the performance can be achieved using different neural network regularization methods such as a bayesian regularization (see BRANN architecture).

## Datasets

## Model

The neural network model consists of two fully-connected layers with ReLU activations and a regression head. The dimensionality of the layers is shown below:

<img src="https://user-images.githubusercontent.com/4588093/72859687-d3ca9580-3d18-11ea-8f28-ff0e89d2940f.png" width="200">

As a loss function we use the mean squared error. The model has been trained using Adam optimizer. The model has been developed using Keras API implemented in Tensorflow.

## Results of evaluation

The squared correlation coefficient for the test set equals 0.73. The example of usage and the evaluation of the trained model is implemented in the script `model_test.py`. To run this script, one has to install all requirements by, for instance, invoking the command: `pip install -r requirements.txt`.

Below is more information on the model performance on the train and test sets.


<div class="row">
  <div class="col-md-8" markdown="1">
<img src="https://user-images.githubusercontent.com/4588093/72859688-d3ca9580-3d18-11ea-82eb-b5919efbe629.png" width="300">
<img src="https://user-images.githubusercontent.com/4588093/72859689-d3ca9580-3d18-11ea-842b-394be8cf4991.png" width="300">
  </div>
  <div class="col-md-4" markdown="1">
<img src="https://user-images.githubusercontent.com/4588093/72859690-d4632c00-3d18-11ea-8d4c-99465d844cc6.png" width="300">
<img src="https://user-images.githubusercontent.com/4588093/72859692-d4632c00-3d18-11ea-8490-a994d2a68315.png" width="300">
  <img height="600px" class="center-block" src="../img/folder/blah.jpg">
  </div>
</div>




## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details



