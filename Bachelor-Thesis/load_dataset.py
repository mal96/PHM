# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:21:35 2018

Function to load dataset.

@author: maliang
"""
import numpy as np


def load_dataset(path):
    x_train = np.load(path+'x_train.npy')
    y_train = np.load(path+'y_train.npy')
    x_test = np.load(path+'x_test.npy')
    y_test = np.load(path+'y_test.npy')
    return (x_train, y_train), (x_test, y_test)

#(x_train, y_train), (x_test, y_test) = load_dataset('dataset/', one_hot_opsettings=True)

