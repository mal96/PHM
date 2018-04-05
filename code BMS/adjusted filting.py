# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:09:18 2017

重构的协同过滤代码

@author: maliang
"""

from scipy.stats import pearsonr
from scipy.spatial import distance
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
import xlwt
import statsmodels.api as sm
import time
from scipy.optimize import curve_fit

time1 = time.time()

'''parameters'''
pre_start_value = 0.9
train_start_value = 0.95
train_end_value = 0.9
fail_value = 0.85
sim_start_value = 0.95
sim_end_value = 0.9
sim_method = 'norm'
order = 2
observe_capacity = 0.85
sample_ratio = 0.2
svd_permit = False
sub_first = False

'''ploy2 to fit data'''
def poly2(x, a, b, c):
    return a*x*x + b*x + c

def extend(seq, extend_ratio):
    add_point_num = int(len(seq) * extend_ratio)
    x = [i for i in range (len(seq) + add_point_num)]
    x_fit = x[:len(seq)]
    x_extend = x[len(seq):]
    popt, pcov = curve_fit(poly2, x_fit, seq, bounds = \
                           ([-np.inf, -np.inf, -np.inf], [0, np.inf, np.inf]))
    y_extend = [poly2(item, *popt) for item in x_extend]
    end_error = seq[-1] - poly2(x_fit[-1], *popt)
    for i in range (len(y_extend)):
        y_extend[i] += end_error
    return y_extend

class battery(object):
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity
        self.length = len(capacity)
        self.point90 = np.where(np.array(self.capacity) < 0.9)[0][0]
        self.point85 = np.where(np.array(self.capacity) < 0.85)[0][0]
        self.sim_start_point = np.where(np.array(self.capacity) < sim_start_value)[0][0]
        self.sim_end_point = np.where(np.array(self.capacity) < sim_end_value)[0][0]
        self.train_start_point = np.where(np.array(self.capacity) < train_start_value)[0][0]
        self.train_end_point = np.where(np.array(self.capacity) < train_end_value)[0][0]
        self.pre_start_point = np.where(np.array(self.capacity) < pre_start_value)[0][0]
        
    def cap_before90(self):
        return self.capacity[:self.point90 + 1]
    def cap_after90(self):
        return self.capacity[self.point90: ]
    def smooth_capacity(self, force):
        return smooth(self.capacity, force)

def smooth(seq, frac):
    x = [i for i in range (len(seq))]
    sm_array = sm.nonparametric.lowess(seq, x, frac = frac, return_sorted=False)
    sm_seq = []
    for i in range (sm_array.shape[0]):
        sm_seq.append(sm_array[i])
    return sm_seq

def cal_similarity(seq1, seq2, method):
    if method == 'pear':
        sim = pearsonr(seq1, seq2)[0]
        return sim
    elif method == 'cos':
        sim = 1 - 0.5*distance.cosine(seq1, seq2)
        return sim
    elif method == 'norm':
        d_norm = distance.norm(np.array(seq1) - np.array(seq2), ord = order)
        sim = 1 / (1 + d_norm)
#        sim = np.exp( - d_norm)
        return sim
    elif method == 'diff':
        _error = []
        _diff = []
        for i in range (len(seq1)):
            _error.append(seq2[i] - seq1[i])
        for i in range (len(_error) - 1):
            _diff.append(abs(_error[i+1] - _error[i]))
        d_diff = np.mean(_diff)
        sim = np.exp( - 10 * d_diff)
        return sim
    else:
        raise Exception('Do not have this kind of sim-method.')
        return -1

prelist_file = xlrd.open_workbook('Prelist.xls')
prelist_sheet = prelist_file.sheet_by_index(0)
prelist = prelist_sheet.col_values(0)
prelist_cut = prelist[:]

del(prelist_file)
del(prelist_sheet)

acc = {}
most_sim = {}
db = {}

for target in prelist:
    t_temp = target.split('_')[0]
    t_group = target.split('_')[1]
    t_bar = target.split('_')[2]
    
    file_nonsm = xlrd.open_workbook('nonsmooth\\' + t_temp + '.xls')
    sheet_nonsm = file_nonsm.sheet_by_name(t_group)
    barlist_nonsm = sheet_nonsm.row_values(0)
    icol = barlist_nonsm.index(t_bar)
    capacity = [item for item in sheet_nonsm.col_values(icol) if isinstance(item, float)]
    capacity.pop(0)
    newbattery = battery(name = target, capacity = capacity)
    db[target] = newbattery

del(barlist_nonsm)
del(capacity)
del(target)
del(t_bar)
del(t_group)
del(t_temp)

process = 1
for pre_code in prelist_cut:
    db_copy = db.copy()
    target = db[pre_code]
    min_length = target.length
    for sample in db:
        '''筛选'''
        if sample == pre_code:#排除自己
            db_copy.pop(sample)
            continue
        if sample.split('_')[0] != pre_code.split('_')[2]:#排除不同温度
            db_copy.pop(sample)
            continue
        if sample.split('_')[2][:9] == pre_code.split('_')[2][:9]:#排除同组
            db_copy.pop(sample)
            continue
        if db[sample].length < target.point90:#排除总长度不及目标0.9的样本
            db_copy.pop(sample)
            continue
        
        t_sim_sm = target.capacity[target.sim_start_point:target.sim_end_point]
        t_sim_sm = smooth(t_sim_sm, 0.2)
        s_sim_sm = db[sample].capacity[target.sim_start_point:target.sim_end_point]
        s_sim_sm = smooth(s_sim_sm, 0.2)
        error = [(t_sim_sm[i] - s_sim_sm[i]) for i in range (len(t_sim_sm))]
        diff = [(error[i+1] - error[i]) for i in range (len(error) - 1)]
        judge_win = int(0.6*len(diff))
        bad_point = 0
        for diff_value in diff:
            if abs(diff_value) > 0.00028:
                bad_point += 1
        if bad_point >= judge_win:#斜率筛选
            db_copy.pop(sample)
            continue
        del(t_sim_sm)
        del(s_sim_sm)
        del(error)
        del(diff)
        del(judge_win)
        del(bad_point)
        '''-----------------'''
        
        if db[sample].length < min_length:
            min_length = db[sample].length
    print(pre_code + 'has' + str(len(db_copy)) + 'co-samples. ('\
          + str(process) + '/' + str(len(prelist_cut)) + ')')
    process += 1
    
    if len(db_copy) == 0:
        acc[pre_code] = ['not enough sample']
        most_sim[pre_code] = ['not enough sample']
        continue
    
    need_svd = True
    if len(db_copy) == 1:#如果只有一个协同样本 则没有svd的必要
        need_svd = False
    
    sim_start_point = 0
    sim_end_point = 0
    target_simpart = target.capacity[target.sim_start_point: target.sim_end_point]
    target_simpart_sm = smooth(target_simpart, frac = 0.2)
    
    sim_dict = {}
    if need_svd and svd_permit:
        matrix2be_svd = np.array(target_simpart_sm)
        for name, sample in db_copy.items():
            next_seq = sample.capacity[target.sim_start_point: target.sim_end_point]
            next_seq = smooth(next_seq, frac = 0.2)
            next_seq = np.array(next_seq)
            matrix2be_svd = np.c_[matrix2be_svd, next_seq]
        
        u, s_vector, v = np.linalg.svd(matrix2be_svd, 0, 1)
        sum_s = sum(s_vector)
        s_len = len(s_vector)
        temp_sum = 0
        svd_keep_num = 0
        for i in range (s_len):
            temp_sum += s_vector[i]
            if temp_sum > 0.9*sum_s:
                svd_keep_num = i
                break
        
        if svd_keep_num < 1:
            svd_keep_num = 1
        need_feature = [s_vector[i] for i in range (svd_keep_num + 1)]
        need_zero = list([0.0]*(s_len - len(need_feature)))
        need_feature.extend(need_zero)
        del(need_zero)
        del(sum_s)
        del(temp_sum)
        del(svd_keep_num)
        s = np.diag(need_feature)
        us = np.dot(u, s)
        feature_denoise = np.dot(us, v)
        del(s)
        del(us)
    else:
        for name, sample in db_copy.items():
            sample_simpart = sample.capacity[target.sim_start_point: target.sim_end_point]
            sample_simpart_sm = smooth(sample_simpart, frac = 0.2)
            similarity = cal_similarity(target_simpart_sm, sample_simpart_sm, method = sim_method)
            sim_dict[name] = similarity
        
    sim_sorted = sorted(sim_dict.items(), key = lambda sim_dict: sim_dict[1], reverse = True)
    
    most_sim[pre_code] = [sim_sorted[0][0]]
    
    '''绝对相似度限制'''
    sample_num = 0
    for sample in sim_sorted:
        if sample[1] > 0.86:
            sample_num += 1
    
    '''比例相似度限制'''
#    sample_num = int(len(sin_sorted) * sample_ratio)

    if sample_num < 1:
        sample_num + 1
    if sample_num > 6:
        sample_num = 6
    
    print(str(sample_num) + 'similar samples selected.')
    similar_list = []
    for i in range (sample_num):
        if sim_sorted[i][1] > 0.6:
            similar_list.append(sim_sorted[i][0])
    if len(similar_list) == 0:
        similar_list.append(sim_sorted[0][0])
    sample_num = len(similar_list)
    
    '''画相似样本 节省时间可注释'''
    fig = plt.figure(dpi = 200)
    plt.plot(target.capacity, color = 'k', lw = 1.4, label = target.name + '(self)')
    for i in range (sample_num):
        sample_name = sim_sorted[i][0]
        sample_sim = sim_sorted[i][1]
        plt.plot(db_copy[sample_name].capacity, lw = 0.7, label = db_copy[sample_name].name\
                 + ' ' + str(round(sample_sim, 3)))
    plt.tight_layout()
    plt.axvline(target.sim_end_point, color = 'r', lw = 0.5, linestyle = '--')
    plt.axvline(target.sim_start_point, color = 'r', lw = 0.5, linestyle = '--')
    plt.axvline(target.point85, color = 'r', lw = 0.5, linestyle = '--')
    plt.legend(loc = 'best', prop = {'size' : 7})
    fig.savefig('similar sample\\' + target.name + '.jpg')
    plt.close(fig)
    
    t_full_capacity = np.array(target.capacity[:min_length])
    s_full_capacity_array = np.array([])
    
    for i in range (sample_num):
        if i == 0:
            s_full_capacity_array = np.array(db_copy[similar_list[i]].capacity[:min_length])
        else:
            s_full_capacity = np.array(db_copy[similar_list[i]].capacity[:min_length])
            s_full_capacity_array = np.c_[s_full_capacity_array, s_full_capacity]
    
    if not sub_first:
        if len(s_full_capacity_array.shape) == 1:
            s_full_capacity_array = np.c_[s_full_capacity_array, s_full_capacity_array]
        
        t_train = t_full_capacity[target.train_start_point: target.train_end_point]
        s_train = s_full_capacity_array[target.train_start_point: target.train_end_point, :]
        
        '''若用平滑数据训练 则此处不要注释'''
        t_train = smooth(t_train, 0.3)
        s_train_sm = np.array([])
        for i in range (s_train.shape[1]):
            if i == 0:
                ext_sample = s_train[:, i]
                ext_sample = smooth(ext_sample, 0.2)
                s_train_sm = ext_sample
            else:
                ext_sample = s_train[:, i]
                ext_sample = smooth(ext_sample, 0.2)
                s_train_sm = np.c_[s_train_sm, ext_sample]
        s_train = s_train_sm
        '''---------------------------'''
        
        '''lasso获得回归系数'''
        lasso_model = LassoCV(max_iter=100000)
        lasso_model.fit(s_train, t_train)
        
        Alpha = lasso_model.alpha_
        opt_lasso_model = Lasso(alpha = Alpha, max_iter = 100000)
        opt_lasso_model.fit(s_train, t_train)
        
        coef = opt_lasso_model.coef_
        intercept = opt_lasso_model.intercept_
        
        lasso_selected = [similar_list[i] for i in range (sample_num)\
                          if coef[i] != 0]
        weight_nonzero = [weight for weight in coef\
                          if weight != 0]
        if len(weight_nonzero) == 1:
            weight_nonzero.append(0.0)
        
        min_pre_len = 1e5
        for sample in lasso_selected:
            if len(db_copy[sample].capacity) < min_pre_len:
                min_pre_len = len(db_copy[sample].capacity)
        
        s_predict = np.array([])
        for i, sample in enumerate(lasso_selected):
            if i == 0:
                s_predict = np.array(db_copy[sample].capacity[target.pre_start_point:min_pre_len])
            else:
                s_predict = np.c_[s_predict, np.array(db_copy[sample].capacity\
                                                      [target.pre_start_point: min_pre_len])]
        if len(s_predict.shape) == 1:
            s_predict = np.c_[s_predict, s_predict]
            
        '''若用平滑数据预测 则此处不要注释'''
        s_predict_sm = np.array([])
        for i in range (s_predict.shape[1]):
            if i == 0:
                ext_sample = s_predict[:, i]
                ext_sample = smooth(ext_sample, 0.4)
                s_predict_sm = ext_sample
            else:
                ext_sample = s_predict[:, i]
                ext_sample = smooth(ext_sample, 0.4)
                s_predict_sm = np.c_[s_predict_sm, ext_sample]
        s_predict = s_predict_sm
        '''--------------------------'''
        
        '''画出被lasso选中的样本'''
        fig = plt.figure(figsize = (16, 9), dpi = 300)
        p_lasso = plt.subplot(1,2,1)
        p_predict = plt.subplot(1,2,2)
        
        p_lasso.plot(target.capacity, color = 'k', lw = 1.4, label = target.name + '(self)')
        for i, sample in enumerate(lasso_selected):
            p_lasso.plot(db_copy[sample].capacity, lw = 0.7, label = db_copy[sample].name\
                         + ' ' + str(round(weight_nonzero[i], 3)))
        p_lasso.axvline(target.pre_start_point, linestyle = '--', lw = 0.7, color = 'r')
        p_lasso.axvline(target.point90, linestyle = '--', lw = 0.7, color = 'r')
        p_lasso.axhline(0.85, linestyle = '--', lw = 0.7, color = 'r')
        p_lasso.legend(loc = 'best', prop = {'size': 7})
        p_lasso.set_title('Sample selected by lasso')
        p_lasso.set_xlabel('Cycle')
        p_lasso.set_ylabel('Capacity')
        p_lasso.axis([0, int(target.point85 * 1.2), 0.8, 1.01])
        
        '''预测'''
#        pre_outcome = opt_lasso_model.predict(s_predict)#自带方法
        
        pre_outcome = np.dot(s_predict, weight_nonzero) + intercept#相关系数 + 截距
        
        pre_full_capacity = target.cap_before90()
        pre_full_capacity.extend(pre_outcome)
        
        '''如果预测结果未到寿 则尝试一次数据延长'''
        need_extend = False
        if min(pre_full_capacity) > 0.85:
            need_extend = True
        if need_extend:
            s_predict_extend = np.array([])
            for i in range (s_predict.shape[1]):
                if i == 0:
                    ext_sample = s_predict[:, i]
                    ext_sample = extend(ext_sample, 3)
                    s_predict_extend = np.array(ext_sample)
                else:
                    ext_sample = s_predict[:, i]
                    ext_sample = extend(ext_sample, 3)
                    s_predict_extend = np.c_[s_predict_extend, np.array(ext_sample)]
            s_predict = np.r_[s_predict, s_predict_extend]
            pre_outcome = np.dot(s_predict, weight_nonzero) + intercept
            pre_full_capacity = target.cap_before90()
            pre_full_capacity.extend(pre_outcome)
            
        '''画结果曲线'''
        p_predict.plot(target.capacity, color = 'k', lw = 1.4, label = 'Original data')
        p_predict.plot(pre_full_capacity, color = 'r', lw = 0.7, label = 'Predict data')
        p_predict.axvline(target.point90, linestyle = '--', lw = 0.7, color = 'r')
        p_predict.axhline(0.85, linestyle = '--', lw = 0.7, color = 'r')
        p_predict.legend(loc = 'best', prop = {'size' : 7})
        p_predict.set_title('Predict outcome')
        p_predict.set_xlabel('Cycle')
        p_predict.set_ylabel('Capacity')
        p_lasso.axis([0, int(target.point85 * 1.2), 0.8, 1.01])
        
        fig.savefig('figure outcome\\' + pre_code + '.jpg')
        
        '''整理结果'''
        if min(pre_full_capacity) > 0.85:
            acc[pre_code] = ['Has not reach fail-point yet']
            continue
        else:
            pre_fail_point = np.where(np.array(pre_full_capacity) < fail_value)[0][0]
            accuracy = 1 - (abs(pre_fail_point - target.point85) / target.point85)
            acc[pre_code] = [accuracy, target.point85, pre_fail_point]
        
        '''结果微调'''
        front_win_seq = target.cap_before90()[-20:]
        front_win_seq = smooth(front_win_seq, 0.4)
        pre_win_seq = pre_outcome[:20]
        pre_win_seq = smooth(pre_win_seq, 0.4)
        diff_front = [abs(front_win_seq[i+1] - front_win_seq[i]) for i in range\
                      (len(front_win_seq) - 1)]
        diff_pre = [abs(pre_win_seq[i+1] - pre_win_seq[i]) for i in range (len(pre_win_seq) - 1)]
        if np.mean(diff_front) < np.mean(diff_pre) - 0.0001:
            pre_fail_point += 50
            continue
        if np.mean(diff_front) < np.mean(diff_pre) - 0.00005:
            pre_fail_point += 20
            continue
        if np.mean(diff_front) > np.mean(diff_pre) + 0.0001:
            pre_fail_point -= 50
            continue
        if np.mean(diff_front) > np.mean(diff_pre) - 0.00005:
            pre_fail_point -= 20
            continue
        accuracy = 1-(abs(pre_fail_point - target.point85) / target.point85)
        acc[pre_code].append(accuracy)
    
    else:
        pass

'''输出结果文件'''
file = xlwt.Workbook()
sheet = file.add_sheet('A')
irow = 0
for t in acc:
    if len(acc[t]) == 1:
        sheet.write(irow, 0, t)
        sheet.write(irow, 1, acc[t][0])
        irow += 1
    elif len(acc[t]) == 3:
        sheet.write(irow, 0, t)
        sheet.write(irow, 1, acc[t][0])
        sheet.write(irow, 2, acc[t][1])
        sheet.write(irow, 3, acc[t][2])
        irow += 1
    else:
        sheet.write(irow, 0, t)
        sheet.write(irow, 1, acc[t][0])
        sheet.write(irow, 2, acc[t][1])
        sheet.write(irow, 3, acc[t][2])
        sheet.write(irow, 4, acc[t][3])
        irow += 1

irow = 0
for t in most_sim:
    sheet.write(irow, 5, most[t][0])
    irow += 1
file.save('outcome.xls')
        
        
        




















