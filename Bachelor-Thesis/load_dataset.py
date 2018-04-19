# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:21:35 2018

Function to load dataset.

@author: maliang
"""
import numpy as np


def load_dataset(path):
    x_train = np.load(path + 'x_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'x_test.npy')
    y_test = np.load(path + 'y_test.npy')
    return (x_train, y_train), (x_test, y_test)


def generate_next_engine(X, y, index):
    end_index = np.where(y == 0)[0][index]
    if index == 0:
        start_index = 0
    else:
        start_index = np.where(y == 0)[0][index - 1] + 1
    X_one_engine = X[start_index:end_index + 1]
    y_one_engine = y[start_index:end_index + 1]
    return X_one_engine, y_one_engine
