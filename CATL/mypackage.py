import statsmodels.api as sm
import time
import numpy as np
from copy import deepcopy
import pandas as pd
import os
from numpy.linalg import norm
from math import acos


# A time machine
# timer.start() at begin
# timer.end() at end, then print total execute time
class timer(object):
    def __init__(self):
        self.time1 = 0
        self.time2 = 0

    def start(self):
        self.time1 = time.time()

    def end(self):
        self.time2 = time.time()
        self.total_time = self.time2 - self.time1
        self.print_string = 'Total execute time: ' + str(
            round(self.total_time, 2)) + 's.'
        print('-' * len(self.print_string))
        print(self.print_string)
        print('-' * len(self.print_string))


# Function:   smooth a time sequence with noise by LOWESS algorithm
#
# Input:      seq:      raw sequence with noise
#             frac:     strength of smooth
#
# Output:     sm_seq:   smoothed sequence with less noise
def smooth(seq, frac):
    x = [i for i in range(len(seq))]
    sm_array = sm.nonparametric.lowess(seq, x, frac=frac, return_sorted=False)
    sm_seq = []
    for i in range(sm_array.shape[0]):
        sm_seq.append(sm_array[i])
    return sm_seq


# Input:         fading sequence before 95 capacity retention (seq)
# Output:        a new sequence containing differential values with scale
#                milestones: all window_bottoms while loops
#
# window_size:   the scale to calculate difference
#                -- 40 in general
#
# stride:        the interval to calculate difference
#                -- 10 in general
#
# need_smooth:   if smooth is not needed, then pass False (default);
#                else, pass a float in range 0-1 to control smooth strength
#                -- e.g. 0.1
def diff_with_scale(seq, window_size, stride, need_smooth=False):
    if len(seq) < window_size + 1:
        raise Exception('Input sequence is to short.')
    if need_smooth == False:
        pass
    else:
        seq = smooth(seq, need_smooth)
    diff_values = list()
    window_bottom = 0
    window_top = window_bottom + window_size
    window_bottoms = list()
    while 1:
        if window_top >= len(seq):
            break
        window_bottoms.append(window_bottom)
        diff_values.append(seq[window_bottom] - seq[window_top])
        window_bottom += stride
        window_top = window_bottom + window_size
    return diff_values, window_bottoms


# Function:   search the point which is the most close to its neighbors
#             in a certain sequence (the most "flat" one)
#
# Input:      a sequence containing many values
#
# Output:     index of the msot "flat" value
def search_flat(seq, rank=1):
    deviation_with_neighbors = list()
    for i, _ in enumerate(seq):
        if i == 0:
            deviation_with_neighbors.append(999)
        elif i == len(seq) - 1:
            deviation_with_neighbors.append(999)
        else:
            deviation_with_neighbors.append(
                max(abs(seq[i] - seq[i - 1]), abs(seq[i] - seq[i + 1])))
        flat_value_sorted = sorted(deviation_with_neighbors)
        flat_value = flat_value_sorted[rank - 1]
        flat_index = deviation_with_neighbors.index(flat_value)
        return flat_index


# Function:   use an absolute window_size to get base value
#
# Input:      fading_data:     raw fading curve
#             window_size:     width of sliding window to calculate difference
#             stride:          interval to calculate difference, every 10 points to generate one value
#             need_smooth:     if don't need smooth, then pass False
#                              if need smooth, then pass a float in range 0-1
#
# Output:     base_value:      base difference value, float
#             window_bottom:   minimun cycle of range to calculate base value
#             window_top:      maximun cycle of range to calculate base value
def get_base_absolute(fading_data, window_size=40, stride=10, need_smooth=0.1):
    cut_off = np.where(fading_data > 0.95)[0][-1]
    bias = np.where(fading_data > 0.99)[0][-1]
    if bias == cut_off:
        bias = cut_off - window_size
    if bias < 0:
        bias = 0
    y_before_95 = fading_data[bias:cut_off]
    if len(y_before_95) <= 70:
        # calculate difference within the last 50% cycles and return as base
        window_bottom = len(y_before_95) - 1 - int(len(y_before_95) / 2)
        window_top = len(y_before_95) - 1
        if need_smooth == False:
            pass
        else:
            y_before_95 = smooth(y_before_95, frac=need_smooth)
        base_value = y_before_95[window_bottom] - y_before_95[window_top]
    else:
        # call Function: diff_with_scale()
        # return a list containing differential values
        # select the value which is the most close to its neighbors as base
        diff_values, window_bottoms = diff_with_scale(
            y_before_95,
            window_size=window_size,
            stride=stride,
            need_smooth=need_smooth)
        rank = 1
        flat_index = search_flat(diff_values, rank=rank)
        base_value = diff_values[flat_index]
        while base_value < 0:
            rank += 1
            flat_index = search_flat(diff_values, rank=rank)
            base_value = diff_values[flat_index]
        window_bottom = window_bottoms[flat_index]
        window_top = window_bottom + window_size
    window_bottom += bias
    window_top += bias
    return base_value, window_bottom, window_top


# Function:   use a relative proportion window_size to get base value
#
# Input:      fading_data:           raw fading curve
#             window_proportion:     proportion of width of sliding window to calculate difference
#                                    and length of .99 to .95
#             stride:                interval to calculate difference, every 10 points to generate one value
#             need_smooth:           if don't need smooth, then pass False
#                                    if need smooth, then pass a float in range 0-1
#
# Output:     base_value:            base difference value, float
#             window_bottom:         minimun cycle of range to calculate base value
#             window_top:            maximun cycle of range to calculate base value
def get_base_proportion(fading_data,
                        window_proportion=0.1,
                        stride=10,
                        need_smooth=0.1):
    cut_off = np.where(fading_data > 0.95)[0][-1]
    bias = np.where(fading_data > 0.99)[0][-1]
    if bias == cut_off:
        bias = cut_off - int(window_proportion * cut_off)
    if bias < 0:
        bias = 0
    y_before_95 = fading_data[bias:cut_off]
    window_size = int(len(y_before_95) * window_proportion)
    if len(y_before_95) <= 70:
        # calculate difference within the last 50% cycles and return as base
        window_bottom = len(y_before_95) - 1 - int(len(y_before_95) / 2)
        window_top = len(y_before_95) - 1
        if need_smooth == False:
            pass
        else:
            y_before_95 = smooth(y_before_95, frac=need_smooth)
        base_value = y_before_95[window_bottom] - y_before_95[window_top]
    else:
        # call Function: diff_with_scale()
        # return a list containing differential values
        # select the value which is the most close to its neighbors as base
        diff_values, window_bottoms = diff_with_scale(
            y_before_95,
            window_size=window_size,
            stride=stride,
            need_smooth=need_smooth)
        rank = 1
        flat_index = search_flat(diff_values, rank=rank)
        base_value = diff_values[flat_index]
        while base_value < 0:
            rank += 1
            flat_index = search_flat(diff_values, rank=rank)
            base_value = diff_values[flat_index]
        window_bottom = window_bottoms[flat_index]
        window_top = window_bottom + window_size
    window_bottom += bias
    window_top += bias
    return base_value, window_bottom, window_top


# Function:   remove abnormal value in a fading data sequence
#
# Input:      seq:              raw fading data sequence
#             error:            threshold.
#                               (Remove a value if it varies from normal value over the threshold)
#
# Output:     index_abnormal:   indexes of abnormal values in raw sequence
#             seq_modified:     modified sequence after removing abnormal values
def modify_abnormal_values(seq, error=0.006):
    index_abnormal = list()
    seq_smooth = smooth(seq, 0.1)
    for i in range(len(seq)):
        if i == 0 or i == len(seq) - 1 or i == 1 or i == len(seq) - 2:
            pass
        else:
            if abs(seq[i] - seq[i - 1]) > error and abs(
                    seq[i] - seq[i + 1]) > error:
                index_abnormal.append(i)
            if abs(seq[i] - seq[i - 2]) > error and abs(
                    seq[i] - seq[i + 2]) > error:
                index_abnormal.append(i)
            if abs(seq[i] - seq_smooth[i]) > 0.7:
                index_abnormal.append(i)
    seq_modified = deepcopy(seq)
    for index in index_abnormal:
        seq_modified[index] = seq_smooth[index]
    if seq_modified[0] < 0.8:
        seq_modified[0] = seq[1]
    return index_abnormal, seq_modified


# Function:   calculate ratio of slope in a fading data sequence
#
# Input:      seq:                      raw fading data sequence, need to fade below .95 already
#             base_window_size:         width of sliding window if use absolute window size
#                                       default: 40
#             base_window_proportion:   width proportion of sliding window if use relative window size
#                                       default: 0.1
#             diff_window_size:         width of sliding window to calculate difference
#                                       default: 40
#             base_stride:              interval to calculate difference when searching for base value
#                                       default: 10
#             diff_stride:              interval to calculate difference when warning in processing
#                                       default: 10
#             remove_abnormal:          whether to remove abnormal values
#                                       Bool. default: True
#             base_use_proportion:      whether to use relative window size when searching for base value
#                                       Bool. default: True
#             epsilon:                  constrant added to denominator and numerator
#
# Output:     window_tops:              milestones of each k-rate value in time-axis
#             k_rate_trend:             ratio of slope trend
def k_rate_trend(seq,
                 base_window_size=40,
                 base_window_proportion=0.1,
                 diff_window_size=40,
                 base_stride=10,
                 diff_stride=10,
                 need_smooth=0.1,
                 remove_abnormal=True,
                 base_use_proportion=True,
                 epsilon=1e-5
                 use_fitting=True):
    if remove_abnormal:
        _, seq = modify_abnormal_values(seq)
    else:
        pass
    if base_use_proportion:
        get_base = get_base_proportion
        second_arg = base_window_proportion
    else:
        get_base = get_base_absolute
        second_arg = base_window_size
    base_value, window_bottom, window_top = get_base(
        seq, second_arg, stride=base_stride, need_smooth=need_smooth)
    diff_values, window_bottoms = diff_with_scale(
        seq,
        window_size=diff_window_size,
        stride=diff_stride,
        need_smooth=need_smooth)
    if use_fitting:
        base_value_scaled = base_value
        diff_values_scaled = diff_values
    else:
        base_value_scaled = base_value / (window_top - window_bottom)
        diff_values_scaled = np.array(diff_values) / diff_window_size
    window_tops = np.array(window_bottoms) + diff_window_size
    k_rate_scaled = (diff_values_scaled + epsilon) / (
        base_value_scaled + epsilon)
    return window_tops, k_rate_scaled


def k_varience_trend(seq,
                     base_window_size=40,
                     base_window_proportion=0.1,
                     diff_window_size=40,
                     base_stride=10,
                     diff_stride=10,
                     need_smooth=0.1,
                     remove_abnormal=True,
                     base_use_proportion=True):
    if remove_abnormal:
        _, seq = modify_abnormal_values(seq)
    else:
        pass
    if base_use_proportion:
        get_base = get_base_proportion
        second_arg = base_window_proportion
    else:
        get_base = get_base_absolute
        second_arg = base_window_size
    base_value, window_bottom, window_top = get_base(
        seq, second_arg, stride=base_stride, need_smooth=need_smooth)
    diff_values, window_bottoms = diff_with_scale(
        seq,
        window_size=diff_window_size,
        stride=diff_stride,
        need_smooth=need_smooth)
    base_value_scaled = base_value / (window_top - window_bottom)
    diff_values_scaled = np.array(diff_values) / diff_window_size
    k_varience_scaled = diff_values_scaled - base_value_scaled
    return window_tops, k_varience_trend


# Function:   extract data from a folder where there are plenty of excel or csv files
#
# Input:      data_path:      folder path
#             column_index:   data location in file
#             file_type:      type of files
#
# Output:     df_sum:         a pandas.DataFrame containing needed data in columns
def extract_fading_data_from_folder(data_path, column_index, file_type):
    file_list = os.listdir(data_path)
    print('Found', str(len(file_list)), 'files in', data_path, '.')
    if file_type == 'excel':
        read_func = pd.read_excel
    elif file_type == 'csv':
        read_func = pd.read_csv
    else:
        raise Exception('Wrong file type.')
    df_sum = pd.DataFrame()
    for file_name in file_list:
        barcode = file_name.split('.')[0].split('_')[1]
        df_1 = read_func(data_path + file_name)
        column_name = df_1.columns[column_index]
        fading_data = np.array(df_1[column_name])
        df = pd.DataFrame()
        df[barcode] = fading_data
        df_sum = pd.concat([df_sum, df], axis=1)
    return df_sum


# Function:   get a straight line between 2 points
#
# Input:      x:    location on x-axis
#             x0:   1st x-coordinate
#             x1:   2nd x-coordinate
#             y0:   1st y-coordinate
#             y1:   2nd y-coordinate
#
# Output:     y:    location on y-axis
def line(x, x0, x1, y0, y1):
    return y0 + (y1 - y0) / (x1 - x0) * (x - x0)


# Function:   search the max error between a curve and the line between its first and last point
#
# Input:      seq_curve:   the curve
#             seq_line:    the line between the first and last point of the input curve
#
# Output:     imax:        location of the max-error point
#             vmax:        value of the max-error
def maxerror(seq_curve, seq_line):
    errorlist = [seq_curve[i] - seq_line[i] for i in range(len(seq_curve))]
    imax = 0
    vmax = 0
    for i, error in enumerate(errorlist):
        if error > vmax:
            vmax = error
            imax = i
    return imax, vmax


def cal_angle(seq):
    seq_sm = seq
    x = [i + 1 for i in range(len(seq_sm))]
    seq_used = seq_sm
    x_used = x
    y = [
        line(x_0, x_used[0], x_used[-1], seq_used[0], seq_used[-1])
        for x_0 in x_used
    ]
    imax, vmax = maxerror(seq_used, y)
    if vmax <= 0:
        return 0
    NextPoint = imax
    while 1:
        NextPoint += 1
        if NextPoint == len(y) - 1:
            break
        if seq_used[NextPoint] <= y[NextPoint]:
            break
        FirstPoint = np.array([x_used[0] / 1000, seq_used[0]])
        UpPoint = np.array([x_used[imax] / 1000, seq_used[imax]])
        EndPoint = np.array([x_used[NextPoint] / 1000, seq_used[NextPoint]])
        DistA = norm(FirstPoint - EndPoint)
        DistB = norm(UpPoint - EndPoint)
        DistC = norm(FirstPoint - UpPoint)
        Cos_C = (DistA**2 + DistB**2 - DistC**2) / (2 * DistA * DistB)
        Rad_C = acos(Cos_C)
        return Rad_C


def angle_trend(seq, xs, drop_beginning=True, need_smooth=0.1):
    if drop_beginning:
        start_point = np.where(seq > 0.99)[0][-1]
    else:
        start_point = 0
    if need_smooth:
        seq = smooth(seq, need_smooth)
        seq = np.array(seq)
    else:
        pass
    milestones = list()
    angles = list()
    for x in xs:
        if x <= start_point + 5:
            continue
        else:
            milestones.append(x)
            cal_part = seq[start_point:x]
            angle = cal_angle(cal_part)
            angles.append(angle)
    return milestones, angles


def limit_maximun(seq, max_value=1.0):
    modified = list()
    for value in seq:
        if value <= max_value:
            modified.append(value)
        else:
            modified.append(max_value)
    return np.array(modified)


def limit_minimun(seq, min_value=0.0):
    modified = list()
    for value in seq:
        if value >= min_value:
            modified.append(value)
        else:
            modified.append(min_value)
    return np.array(modified)


def to_warning_level(indicator1, indicator2, indicator3, scale1, scale2,
                     scale3, weight1, weight2, weight3):
    if type(indicator1) != np.ndarray:
        indicator1 = np.array(indicator1)
    if type(indicator2) != np.ndarray:
        indicator2 = np.array(indicator2)
    if type(indicator3) != np.ndarray:
        indicator3 = np.array(indicator3)

    indicator1 /= scale1
    indicator2 /= scale2
    indicator3 /= scale3

    indicator1 = limit_maximun(indicator1)
    indicator2 = limit_maximun(indicator2)
    indicator3 = limit_maximun(indicator3)

    indicator1 = limit_minimun(indicator1)
    indicator2 = limit_minimun(indicator2)
    indicator3 = limit_minimun(indicator3)
    prob = [
        indicator1[i] * weight1 + indicator2[i] * weight2 +
        indicator3[i] * weight3 for i in range(len(indicator1))
    ]
    return np.array(prob)
