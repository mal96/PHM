import pandas as pd
import numpy as np
import os


def limit_maximum(seq, max_value):
    modified = list()
    for value in seq:
        if value <= max_value:
            modified.append(value)
        else:
            modified.append(max_value)
    return np.array(modified)


Memory_Len = 5
Step = Memory_Len  # If no overlapping between each sample
# Step = 1
Limit_RUL = False
Net_Structure = 'LSTM'
Save_Path = 'dataset_no_overlapping_' + Net_Structure + '/'  # If no overlapping between each sample
# Save_Path = 'dataset_' + Net_Structure + '/'
data_path = 'data_with_engine_ID/'
need_smooth = True

if need_smooth:
    tail = '_smooth.csv'
    sub_save_path = 'smooth/'
else:
    tail = '_transformed.csv'
    sub_save_path = 'normalized/'

training_list = pd.read_csv('training_list.csv', header=None)
testing_list = pd.read_csv('testing_list.csv', header=None)
training_list = np.array(training_list.iloc[:, 0])
testing_list = np.array(testing_list.iloc[:, 0])

engine_list = [i + 1 for i in range(218)]

training_set = list()
training_label = list()
testing_set = list()
testing_label = list()

## Load training data, convert to training set
for engine_ID in engine_list:
    print('Engine ID:', str(engine_ID))  # print log
    df_engine = pd.read_csv(data_path + str(engine_ID) + tail)
    data_matrix = np.array(df_engine.iloc[:, 1:])
    life = df_engine.shape[0]
    RUL_vector = np.array([life - i - 1 for i in range(life)])

    if Limit_RUL:
        RUL_vector = limit_maximum(RUL_vector, Limit_RUL)
    else:
        pass

    top = life
    bottom = top - Memory_Len
    while bottom >= 0:
        slice_matrix = data_matrix[bottom:top, :]
        if Net_Structure == 'LSTM':
            pass
        else:
            slice_matrix = slice_matrix.T
        if int(engine_ID) in training_list:
            training_set.insert(
                0, slice_matrix
            )  # insert slice matrix to the first situation of list
            # training_set.append(slice_matrix)
            training_label.insert(0, RUL_vector[top - 1])
            # training_label.append(RUL_vector[top - 1])
        else:
            testing_set.insert(0, slice_matrix)
            # testing_set.append(slice_matrix)
            testing_label.insert(0, RUL_vector[top - 1])
            # testing_label.append(RUL_vector[top - 1])
        top -= Step  # If no overlapping between each sample
        # top -= 1
        bottom = top - Memory_Len

training_set = np.array(training_set)
testing_set = np.array(testing_set)
training_label = np.array(training_label)
testing_label = np.array(testing_label)

if os.path.isdir(Save_Path):
    pass
else:
    os.mkdir(Save_Path)

np.save(Save_Path + sub_save_path + 'x_train', training_set)
np.save(Save_Path + sub_save_path + 'y_train', training_label)
np.save(Save_Path + sub_save_path + 'x_test', testing_set)
np.save(Save_Path + sub_save_path + 'y_test', testing_label)
