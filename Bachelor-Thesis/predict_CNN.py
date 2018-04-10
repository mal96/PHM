# CNN训练RMSE: 25.952
# CNN测试RMSE: 42.245


import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from load_dataset import load_dataset

# Load data...
(x_train, y_train), (x_test, y_test) = load_dataset('dataset_CNN/')
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


# Define some constants
MEMORY_LEN = x_train.shape[2]
FEATURES_NUM = x_train.shape[1]
batch_size = 64
n_epochs = 100


def build_model():
    model = Sequential()
    # Input shape: (FEATURES_NUM, 25, 1)
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(3, 6),
            padding='valid',
            strides=1,
            activation='relu',
            input_shape=(FEATURES_NUM, MEMORY_LEN, 1)))
    # Output shape: (18, 20, 16)
    model.add(
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Dropout(0.1))
    # Output shape: (9, 10, 16)
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(2, 2),
            padding='same',
            strides=1,
            activation='relu'))
    # Output shape: (9, 10, 32)
    model.add(MaxPooling2D(pool_size=(3, 2), strides=(3, 2), padding='valid'))
    model.add(Dropout(0.3))
    # Output shape: (3, 5, 32)
    model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=1, padding='valid', activation='relu'))
    # Output shape: (2, 4, 32)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    # Output shape: (1, 2, 32)
    model.add(Flatten())
    # Output shape: (64, )
    model.add(Dense(units=32, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model

optimizer = 'RMSProp'
model = build_model()
model.compile(optimizer=optimizer, loss='mse')
model.summary()
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=128,
    epochs=n_epochs,
    verbose=1,
    validation_data=(x_test, y_test))
y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
print('Test mean squared error:', str(round(mse, 2)))

fig = plt.figure(figsize=(9, 6))
plt.plot(history.history['loss'], '--.', label='loss', lw=1.2)
plt.plot(history.history['val_loss'], '-^', label='val loss', lw=1.2)
plt.xlabel('Epoch', size=20)
plt.ylabel('Loss', size=20)
plt.title('CNN Training Process', size=20)
plt.xlim(0, 99)
plt.legend(prop={'size':20})
plt.savefig('loss_'+optimizer+'.jpg', dpi=500)
plt.clf()
plt.close()

plot_model(model, to_file='CNN.png')


def rmse(y_pred, y_true):
    n = len(y_pred)
    result = sum([(value_pred-value_true)**2 for value_pred, value_true in zip(y_pred, y_true)])
    result = result / n
    result = np.sqrt(result)
    return result
