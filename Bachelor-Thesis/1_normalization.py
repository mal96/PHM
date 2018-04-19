'''
Load data from folder 'data_with_engine_ID/';
Split to train and test sets;
Encode operate settings to one-hot key;
Remove insensitive sensors;
Limit maximun RUL within a certain value (selectable)
'''
import os

import numpy as np
import pandas as pd
from numpy.random import rand
from sklearn.cluster import KMeans


# Split engines into training and testing set. Once finished, then just read results from csv.
'''
def train_test_split(engine_list, training_proportion):
    if type(engine_list) == list:
        engine_list = np.array(engine_list)
    else:
        pass
    index = rand(len(engine_list)) < training_proportion
    training_list = engine_list[index]
    testing_list = engine_list[~index]
    return training_list, testing_list

engine_list = [str(i + 1) for i in range(218)]
training_list, testing_list = train_test_split(engine_list, training_proportion=0.8)
'''


class Min_Max_Scaler(object):
    def __init__(self, bottom, top):
        self.bottom = bottom
        self.top = top

    def fit(self, seq):
        self.maximum = max(seq)
        self.minimum = min(seq)

    def predict(self, seq):
        seq_transformed = list()
        for value in seq:
            seq_transformed.append(self.bottom + (self.top - self.bottom) * (
                value - self.minimum) / (self.maximum - self.minimum))
        return seq_transformed

    def predict_scalar(self, value):
        return self.bottom + (self.top - self.bottom) * (
            value - self.minimum) / (self.maximum - self.minimum)


def filt_certain_regime_data(data_seq, regime_seq, regime_label):
    filted_data = list()
    for data_value, regime_value in zip(data_seq, regime_seq):
        if regime_value == regime_label:
            filted_data.append(data_value)
        else:
            pass
    return filted_data


def to_one_hot(seq, num_class):
    encoded_matrix = np.zeros((1, num_class))
    for value in seq:
        stacked = np.zeros((1, num_class))
        stacked[0, value] = 1
        encoded_matrix = np.r_[encoded_matrix, stacked]
    encoded_matrix = encoded_matrix[1:, :]
    return encoded_matrix


# Read training and testing list from csv.
training_list = pd.read_csv('training_list.csv', header=None)
testing_list = pd.read_csv('testing_list.csv', header=None)
training_list = np.array(training_list.iloc[:, 0])
testing_list = np.array(testing_list.iloc[:, 0])

## Data prepare
data_path = 'data_with_engine_ID/'
insensitive_sensors = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
memory_features = [
    'setting1', 'setting2', 'setting3', 's2', 's3', 's4', 's7', 's8', 's9',
    's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21'
]
# Establish an empty database
training_db = dict()
for feature_name in memory_features:
    training_db[feature_name] = []
# Load data into database
for engine_ID in training_list:
    df_engine = pd.read_csv(data_path + str(engine_ID) + '.csv')
    for i in range(df_engine.shape[1]):
        feature_name = df_engine.columns[i]
        if feature_name in memory_features:
            seq = list(df_engine[feature_name])
            training_db[feature_name].extend(seq)

## Use operate settings to train a cluster model
# Scale setting data to range 0-1
settings_scaler_db = dict()
setting1 = training_db['setting1']
setting2 = training_db['setting2']
setting3 = training_db['setting3']
mms1 = Min_Max_Scaler(bottom=-1, top=1)
mms1.fit(setting1)
settings_scaler_db['setting1'] = mms1  # Save scaler for setting1
setting1 = mms1.predict(setting1)
mms2 = Min_Max_Scaler(bottom=-1, top=1)
mms2.fit(setting2)
settings_scaler_db['setting2'] = mms2  # Save scaler for setting2
setting2 = mms2.predict(setting2)
mms3 = Min_Max_Scaler(bottom=-1, top=1)
mms3.fit(setting3)
settings_scaler_db['setting3'] = mms3  # Save scaler for setting3
setting3 = mms3.predict(setting3)
operate_settings = [setting1, setting2, setting3]
operate_settings = np.array(operate_settings).T
# Train a cluster model to set label for operate settings
cluster_model = KMeans(n_clusters=6)
encoded_operate_settings = cluster_model.fit_predict(operate_settings)

## Train scale model for 14 sensors in each regime
sensor_list = [
    's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15',
    's17', 's20', 's21'
]
scaler_db = dict()
for sensor_name in sensor_list:
    raw_seq = training_db[sensor_name]
    for regime_label in range(6):
        model_key = (sensor_name, regime_label)
        filted_seq = filt_certain_regime_data(
            data_seq=raw_seq,
            regime_seq=encoded_operate_settings,
            regime_label=regime_label)
        mms = Min_Max_Scaler(bottom=-1, top=1)
        mms.fit(filted_seq)
        scaler_db[model_key] = mms

## Load all csv files;
## Encode operate settings into one-hot key;
## Remove insensitive features;
## Scale for each regime;
## (Limit maximum RUL within a certain value)
engine_list = [str(i + 1) for i in range(218)]  # All engines
for engine_ID in engine_list:
    df_engine = pd.read_csv(data_path + engine_ID + '.csv')

    # df_transformed: save transformed data and save to csv
    df_transformed = pd.DataFrame(
        index=[i + 1 for i in range(df_engine.shape[0])])  # Cycle number

    # Encode operate settings
    setting1 = list(df_engine['setting1'])
    setting1 = settings_scaler_db['setting1'].predict(setting1)
    setting2 = list(df_engine['setting2'])
    setting2 = settings_scaler_db['setting2'].predict(setting2)
    setting3 = list(df_engine['setting3'])
    setting3 = settings_scaler_db['setting3'].predict(setting3)
    operate_settings = [setting1, setting2, setting3]
    operate_settings = np.array(operate_settings).T
    encoded_operate_settings = cluster_model.predict(
        operate_settings)  ## 0, 1, 2, 3, 4, 5
    one_hot_operate_settings = to_one_hot(
        encoded_operate_settings, num_class=6)  # One-hot
    # Save one-hot settings to a new DataFrame
    settings_df = pd.DataFrame(
        one_hot_operate_settings,
        index=[i + 1 for i in range(df_engine.shape[0])])

    # Scale sensor data for each regime
    for sensor_name in sensor_list:
        raw_seq = list(df_engine[sensor_name])
        transformed_seq = list()
        for data_value, regime_label in zip(raw_seq, encoded_operate_settings):
            transformed_value = scaler_db[(
                sensor_name, regime_label)].predict_scalar(data_value)
            transformed_seq.append(transformed_value)
        transfromed_seq = np.array(transformed_seq)
        df_transformed[sensor_name] = transformed_seq

    # Stack sensor data and operate settings
    df_transformed = pd.concat([df_transformed, settings_df], axis=1)
    df_transformed.to_csv(data_path + engine_ID + '_transformed.csv')
