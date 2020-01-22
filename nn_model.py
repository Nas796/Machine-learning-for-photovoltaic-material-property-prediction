import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


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


inputs = keras.Input(shape=(632,), name='img')
x = layers.Dense(64, activation='relu', kernel_initializer='normal',
                 kernel_regularizer=regularizers.l2(0.002),
                 activity_regularizer=regularizers.l1(0.002))(inputs)
# x = layers.Dense(16, activation='relu', kernel_initializer='normal')(x)
x = layers.Dense(8, activation='relu', kernel_initializer='normal')(x)
outputs = layers.Dense(1, kernel_initializer='normal')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='model')
model.summary()
keras.utils.plot_model(model, 'my_first_model.png', show_shapes=True)

# load dataset

train = exel2dict('train.xlsx')
test = exel2dict('test.xlsx')

x_train, y_train = exel2numpy('train.xlsx')
x_test, y_test = exel2numpy('test.xlsx')

# norm = np.max(x_train)
# x_train = x_train.astype('float32') / norm
# x_test = x_test.astype('float32') / norm

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype('float32') / 255
# x_test = x_test.reshape(10000, 784).astype('float32') / 255

# model.compile(loss='mean_squared_error',
#               optimizer=keras.optimizers.Adam(),
#               metrics=[tf.keras.metrics.MeanSquaredError()])

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(),
              metrics=[correlation_coefficient])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=100,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard_callback])

# test_scores = model.evaluate(x_test, y_test, verbose=2)
# print('Test loss:', test_scores[0])
# print('Test :', test_scores[1])
