import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xlrd
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV


# Calculate similarity between 2 sequences
def cal_similarity(seq1, seq2, method):
    if method == 'pear':
        sim = pearsonr(seq1, seq2)[0]
        return sim
    elif method == 'cos':
        sim = 1 - 0.5 * distance.cosine(seq1, seq2)
        return sim
    elif method == 'norm':
        d_norm = distance.norm(np.array(seq1) - np.array(seq2), ord=2)
        sim = 1 / (1 + d_norm)
        return sim
    elif method == 'diff':
        _error = []
        _diff = []
        for i in range(len(seq1)):
            _error.append(seq2[i] - seq1[i])
        for i in range(len(_error) - 1):
            _diff.append(abs(_error[i + 1] - _error[i]))
        d_diff = np.mean(_diff)
        sim = np.exp(-10 * d_diff)
        return sim
    else:
        return -1


# Smooth Function
def smooth(seq, frac):
    x = [i for i in range(len(seq))]
    sm_array = sm.nonparametric.lowess(seq, x, frac=frac, return_sorted=False)
    sm_seq = []
    for i in range(sm_array.shape[0]):
        sm_seq.append(sm_array[i])
    sm_seq = np.array(sm_seq)
    return sm_seq


# Select and remove sample in database
def filt(target_name, target_seq, database):
    target_group = target_name.split('_')[2][:9]
    target_point90 = np.where(target_seq > 0.9)[0][-1]
    db_copy = database.copy()
    for samplename in database:
        # remove itself
        if samplename == target_name:
            db_copy.pop(samplename)
            continue
        # remove different temperature
        if samplename.split('_')[0] != target_name.split('_')[0]:
            db_copy.pop(samplename)
            continue
        # remove the same group
        if samplename.split('_')[2][:9] == target_group:
            db_copy.pop(samplename)
            continue
        # remove those whose length is less than target's length 0.9
        if len(database[samplename]) < target_point90:
            db_copy.pop(samplename)
            continue
    return db_copy


# Get all sheets' name in one excel
def get_sheet_names(excel_path):
    excel = xlrd.open_workbook(excel_path)
    sheet_names = excel.sheet_names()
    return sheet_names


# Read data from local format
def read_local_data(path):
    db_dict = dict()
    file_list = os.listdir(path)
    for file_name in file_list:
        temperature = file_name.split('.')[0]
        sheet_names = get_sheet_names(path + file_name)
        for sheetname in sheet_names:
            group = sheetname
            df = pd.read_excel(path + file_name, sheetname=sheetname)
            for icol in range(df.shape[1]):
                barcode = df.columns[icol]
                fading_data = np.array(df[barcode][~df[barcode].isnull()])
                battery_name = temperature + '_' + group + '_' + barcode
                db_dict[battery_name] = fading_data
    return db_dict


# Cut target seq and sample seq, in order to let them have the same n-rows
def cut_slices(target_seq,
               db_sample,
               start_cap,
               end_cap,
               queue,
               target_known_value=0.9,
               need_smooth=0.1):
    # deal with smoothness
    if need_smooth:
        target_known_part = target_seq[:np.where(
            target_seq >= target_known_value)[0][-1]]
        target_known_part = smooth(target_known_part, frac=need_smooth)
        for index in range(len(target_known_part)):
            target_seq[index] = target_known_part[
                index]  # smooth the known part of target sequence
        for samplename in db_sample:
            db_sample[samplename] = smooth(
                db_sample[samplename], frac=need_smooth)  # smooth samples
    else:
        pass
    # cut target sequence
    start_cycle = np.where(target_seq <= start_cap)[0][0]
    end_cycle = np.where(target_seq >= end_cap)[0][-1]
    target_slice = target_seq[start_cycle:end_cycle]
    # cut all samples sequence
    sample_slice_matrix = np.zeros(
        (len(target_slice),
         1))  # create an empty matrix containing one zero-column
    for samplename in queue:
        fading_data = db_sample[samplename]
        sample_slice = fading_data[start_cycle:end_cycle].reshape(
            len(target_slice), 1)
        sample_slice_matrix = np.c_[sample_slice_matrix, sample_slice]
    sample_slice_matrix = sample_slice_matrix[:,
                                              1:]  # delete the first zero-column
    return target_slice, sample_slice_matrix


def get_predict_set(db_sample, queue, start_cycle, end_cycle, need_smooth=0.1):
    if need_smooth:
        for samplename in db_sample:
            db_sample[samplename] = smooth(db_sample[samplename], need_smooth)
    else:
        pass
    sample_slice_matrix = np.zeros((end_cycle - start_cycle, 1))
    for samplename in queue:
        fading_data = db_sample[samplename]
        sample_slice = fading_data[start_cycle:end_cycle].reshape(
            end_cycle - start_cycle, 1)
        sample_slice_matrix = np.c_[sample_slice_matrix, sample_slice]
    sample_slice_matrix = sample_slice_matrix[:, 1:]
    return sample_slice_matrix


def batch_similarity(target_seq,
                     db_sample,
                     sim_start_cap,
                     sim_end_cap,
                     method,
                     target_known_value=0.9,
                     need_smooth=0.1):
    sim_start_cycle = np.where(target_seq <= sim_start_cap)[0][0]
    sim_end_cycle = np.where(target_seq >= sim_end_cap)[0][-1]
    # deal with smoothness
    if need_smooth:
        target_known_part = target_seq[:np.where(
            target_seq >= target_known_value)[0][-1]]
        target_known_part = smooth(target_known_part, frac=need_smooth)
        for index in range(len(target_known_part)):
            target_seq[index] = target_known_part[
                index]  # smooth the known part of target sequence
        for samplename in db_sample:
            db_sample[samplename] = smooth(
                db_sample[samplename], frac=need_smooth)  # smooth samples
    else:
        pass
    target_sim_part = target_seq[sim_start_cycle:sim_end_cycle]
    sim_dict = dict()  # Dict to save similarity value of each sample
    for samplename in db_sample:
        fading_data = db_sample[samplename]
        sample_sim_part = fading_data[sim_start_cycle:sim_end_cycle]
        sim_value = cal_similarity(
            target_sim_part, sample_sim_part, method=method)
        sim_dict[samplename] = sim_value
    return sim_dict


def build_lasso(x_train, y_train):
    model = LassoCV(max_iter=100000)
    model.fit(x_train, y_train)
    return model


# Predict Function
def predict(target_name, target_seq, database, sim_start_cap, sim_end_cap,
            train_start_cap, train_end_cap, pred_start_cap, pred_end_cap,
            sim_method, sample_proportion):
    db_copy = database.copy()
    db_copy = filt(target_name, target_seq, db_copy)
    sample_all = [key for key in db_copy]
    sim_dict = batch_similarity(
        target_seq, db_copy, sim_start_cap, sim_end_cap, method=sim_method)
    sim_sorted = sorted(
        sim_dict.items(), key=lambda x: x[1],
        reverse=True)  # obtain similarity ranking
    num_reserve = int(sample_proportion *
                      len(sim_sorted))  # number of samples to train lasso
    if num_reserve <= 1:
        num_reserve = 1
    if num_reserve >= 6:
        num_reserve = 6
    sim_sorted = sim_sorted[:num_reserve]  # reserve top rankings in sim_sorted
    sample_reserve = [item[0] for item in sim_sorted
                      ]  # the queue to get train matrix in column order
    for samplename in sample_all:
        if samplename not in sample_reserve:
            db_copy.pop(samplename)  # reserve top rankings in db_copy
    target_train, sample_train = cut_slices(
        target_seq,
        db_copy,
        train_start_cap,
        train_end_cap,
        queue=sample_reserve)
    model = build_lasso(sample_train, target_train)
    sample_zero_weight = [
        sample_reserve[i] for i in range(len(sample_reserve))
        if abs(model.coef_[i] < 1e-3)
    ]  # samples which have a zero-weight according lasso model
    min_len = min([
        len(db_copy[samplename]) for samplename in sample_reserve
        if samplename not in sample_zero_weight
    ])  # the minimum length of samples which have a non-zero-weight
    for samplename in sample_zero_weight:
        db_copy[samplename] = np.zeros((min_len, ))
    x_predict = get_predict_set(
        db_copy,
        sample_reserve,
        start_cycle=np.where(target_seq >= pred_start_cap)[0][-1],
        end_cycle=min_len)
    y_predict = model.predict(x_predict)
    y_predict += (
        pred_start_cap - y_predict[0]
    )  # modify the beginning error between pred and true sequence
    # splice true part and pred part
    pred_seq = target_seq[:np.where(target_seq >= pred_start_cap)[0][-1]]
    pred_seq = pred_seq.reshape(len(pred_seq), 1)
    pred_seq = np.r_[pred_seq, y_predict.reshape(len(y_predict), 1)]
    pred_seq = pred_seq.reshape(pred_seq.shape[0], )
    if min(pred_seq) < pred_end_cap:
        pred_life = np.where(pred_seq > pred_end_cap)[0][-1]
    else:
        pred_life = -1
    return pred_seq, pred_life


def main():
    from matplotlib import pyplot as plt
    sim_start_cap = 0.97
    sim_end_cap = 0.9
    train_start_cap = 0.97
    train_end_cap = 0.9
    pred_start_cap = 0.9
    pred_end_cap = 0.85
    sim_method = 'norm'
    sample_proportion = 0.2

    database = read_local_data('P37_data/')
    true_lifes = list()
    pred_lifes = list()
    target_names = list()
    for target_name in database:
        target_seq = database[target_name]
        true_life = np.where(target_seq > pred_end_cap)[0][-1]
        pred_seq, pred_life = predict(
            target_name,
            target_seq,
            database,
            sim_start_cap=sim_start_cap,
            sim_end_cap=sim_end_cap,
            train_start_cap=train_start_cap,
            train_end_cap=train_end_cap,
            pred_start_cap=pred_start_cap,
            pred_end_cap=pred_end_cap,
            sim_method=sim_method,
            sample_proportion=sample_proportion)
        target_names.append(target_name)
        true_lifes.append(true_life)
        pred_lifes.append(pred_life)
        plt.figure(figsize=(5, 3.6))
        plt.plot(target_seq, color='k', lw=1.0, label='True')
        plt.plot(pred_seq, color='r', lw=1.0, label='Pred')
        plt.title('Pred result of ' + target_name, size=16)
        plt.xlabel('Cycle No.', size=14)
        plt.ylabel('Cap. Retention', size=14)
        plt.grid(alpha=0.3)
        plt.axhline(pred_end_cap, lw=0.8, linestyle='--', color='k')
        plt.axvline(
            np.where(target_seq > pred_start_cap)[0][-1],
            lw=0.8,
            linestyle='--',
            color='k')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('outcome/' + target_name + '.jpg', dpi=200)
        plt.clf()
        plt.close()
        print(target_name, 'finished.')
    result = pd.DataFrame(index=target_names)
    result['Truelife'] = true_lifes
    result['Predlife'] = pred_lifes
    result.to_csv('result.csv')


if __name__ == "__main__":
    main()