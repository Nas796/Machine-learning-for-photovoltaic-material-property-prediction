import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


# a set of auxiliary functions to read dataset

def exel2dict(filename):
    xls = pd.ExcelFile(filename)
    df = xls.parse(xls.sheet_names[0])
    data = df.to_dict()

    for key, value in data.items():
        data[key] = np.array(list(value.values()))

    return data


def exel2numpy(filename):
    xls = pd.ExcelFile(filename)
    df = xls.parse(xls.sheet_names[0])
    data = df.to_dict()

    sortednames = sorted(data.keys(), key=lambda x: x.lower())

    data_x = []
    data_y = []

    for key in sortednames:
        if key == 'PCE':
            data_y.append(np.array(list(data[key].values())))
        else:
            data_x.append(np.array(list(data[key].values())))

    return np.array(data_x).T, np.array(data_y).T


# function computing metric based on the squared correlation coefficient

def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)


def main():

    # define model architecture

    inputs = keras.Input(shape=(632,), name='img')
    x = layers.Dense(32, activation='relu', kernel_initializer='normal')(inputs)
    x = layers.Dense(4, activation='relu', kernel_initializer='normal')(x)
    outputs = layers.Dense(1, kernel_initializer='normal')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='model')
    model.summary()
    keras.utils.plot_model(model, 'my_first_model.png', show_shapes=True)

    # load dataset

    x_train, y_train = exel2numpy('train.xlsx')
    x_test, y_test = exel2numpy('test.xlsx')

    # initialize the model

    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(),
                  metrics=[correlation_coefficient])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # train and evaluate the model

    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=100,
                        validation_data=(x_test, y_test),
                        callbacks=[tensorboard_callback])


if __name__ == '__main__':

    main()
