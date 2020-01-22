"""
This script is for testing the trained model named model.h5.
"""
from tensorflow import keras
from model_train import exel2numpy
from model_train import correlation_coefficient


x_test, y_test = exel2numpy('test.xlsx')
new_model = keras.models.load_model('model.h5', custom_objects={"correlation_coefficient": correlation_coefficient})
new_predictions = new_model.predict(x_test)

test_scores = new_model.evaluate(x_test, y_test, verbose=2)
