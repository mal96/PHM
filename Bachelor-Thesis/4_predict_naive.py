# batch size 64, dense 1, adam, val RMSE 42.092, train RMSE 7.713

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:40:20 2018

@author: maliang
"""
from load_dataset import load_dataset
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
import numpy as np

def rmse(y_pred, y_true):
    n = len(y_pred)
    result = sum([(value_pred-value_true)**2 for value_pred, value_true in zip(y_pred, y_true)])
    result = result / n
    result = np.sqrt(result)
    return result


(x_train, y_train), (x_test, y_test) = load_dataset('dataset_LSTM/')


def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.4))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.6))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    return model
optimizer='RMSProp'
model = build_model()
model.compile(optimizer=optimizer, loss='mse')
model.summary()
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=100,
    verbose=1,
    validation_data=(x_test, y_test))
y_predict = model.predict(x_test)
train_predict = model.predict(x_train)
test_rmse = rmse(y_predict, y_test)
train_rmse = rmse(train_predict, y_train)

fig = plt.figure(figsize=(9, 6))
plt.plot(history.history['loss'], '--.', label='loss', lw=1.2)
plt.plot(history.history['val_loss'], '-^', label='val loss', lw=1.2)
plt.xlabel('Epoch', size=20)
plt.ylabel('Loss', size=20)
plt.title('LSTM Training Process', size=20)
plt.xlim(0, 99)
plt.legend(prop={'size':20})
plt.savefig('loss1_'+optimizer+'.jpg', dpi=500)
plt.clf()
plt.close()

from keras.utils import plot_model
plot_model(model, to_file='LSTM.png')







