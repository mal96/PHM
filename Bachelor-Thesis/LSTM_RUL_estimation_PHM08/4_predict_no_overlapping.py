from load_dataset import load_dataset
from load_dataset import generate_next_engine
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 


def rmse(y_pred, y_true):
    n = len(y_pred)
    result = sum([(value_pred - value_true)**2
                  for value_pred, value_true in zip(y_pred, y_true)])
    result = result / n
    result = np.sqrt(result)
    return result


(x_train, y_train), (x_test, y_test) = load_dataset('dataset_no_overlapping_LSTM/smooth/')


def build_model():
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            # input_shape=(x_train.shape[1], x_train.shape[2]),
            batch_input_shape=(1, x_train.shape[1], x_train.shape[2]),
            stateful=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=False, stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    return model


optimizer = 'adam'
n_sample_epoch = 5  # Epoch to train on each engine
n_all_epoch = 500  # Epoch to train on all engines
n_train = 176
n_test = 42

model = build_model()
model.compile(optimizer=optimizer, loss='mse')
# model.summary()   # observe Net structure

rmse_collector = dict()
rmse_collector['train'] = list()
rmse_collector['test'] = list()
for all_epoch in range(n_all_epoch):  # Loop to train on all samples
    for engine_index in range(n_train):  # Loop to traverse each sample
        X_train_one_engine, y_train_one_engine = generate_next_engine(
            x_train, y_train, engine_index)
        for sample_epoch in range(n_sample_epoch):  # Loop to train on each sample
            print('Global epoch:' + str(all_epoch + 1) + ', Sample #' +
                  str(engine_index + 1) + ', Local epoch:' +
                  str(sample_epoch + 1))  # print log
            model.fit(
                X_train_one_engine,
                y_train_one_engine,
                batch_size=1,
                epochs=1,
                verbose=0,
                shuffle=False)  # Train on one sample for once
            model.reset_states()  # Clear conditions

    ## Calculate loss/rmse on training and testing set
    # ---Training set
    RUL_pred = np.array([])
    for engine_index in range(n_train):
        X_train_one_engine, _ = generate_next_engine(x_train, y_train,
                                                     engine_index)
        RUL_pred_one_engine = list()
        for step in range(len(X_train_one_engine)):
            X_train_one_step = X_train_one_engine[step]
            X_train_one_step = X_train_one_step.reshape(1, X_train_one_step.shape[0], X_train_one_step.shape[1])
            RUL_pred_one_step = model.predict(
                X_train_one_step, batch_size=1)
            RUL_pred_one_engine.append(RUL_pred_one_step[0])
        model.reset_states()  # Clear conditions after predicting on one engine
        RUL_pred_one_engine = np.array(RUL_pred_one_engine).reshape(len(X_train_one_engine, ))
        RUL_pred = np.concatenate((RUL_pred, RUL_pred_one_engine))
    rmse_train = rmse(RUL_pred, y_train)
    rmse_collector['train'].append(rmse_train)
    # ---Testing set
    RUL_pred = np.array([])
    for engine_index in range(n_test):
        X_test_one_engine, _ = generate_next_engine(x_test, y_test,
                                                    engine_index)
        RUL_pred_one_engine = list()
        for step in range(len(X_test_one_engine)):
            X_test_one_step = X_test_one_engine[step]
            X_test_one_step = X_test_one_step.reshape(1, X_test_one_step.shape[0], X_test_one_step.shape[1])
            RUL_pred_one_step = model.predict(
                X_test_one_step, batch_size=1)
            RUL_pred_one_engine.append(RUL_pred_one_step[0])
        model.reset_states()  # Clear conditions after predicting on one engine
        RUL_pred_one_engine = np.array(RUL_pred_one_engine).reshape(len(X_test_one_engine, ))
        RUL_pred = np.concatenate((RUL_pred, RUL_pred_one_engine))
    rmse_test = rmse(RUL_pred, y_test)
    rmse_collector['test'].append(rmse_test)

    # print log
    print('--------------------')
    print('Global epoch:'+str(all_epoch+1))
    print('Training rmse = '+str(round(rmse_train, 4)))
    print('Testing rmse = '+str(round(rmse_test, 4)))
    print('--------------------')

    # Save model regularly
    if all_epoch % 30 == 0:
        model.save('well_trained_models/model.h5')
        df_loss = pd.DataFrame()
        df_loss['train'] = rmse_collector['train']
        df_loss['test'] = rmse_collector['test']
        df_loss.to_csv('loss.csv', index=None)
