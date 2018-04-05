# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:08:26 2017
跳水预警算法包，包括：

AngleTrend(seq, startpoint=10, begin_at_top=True):
    '''
    Input:
        seq:需要分析的电池退化数据，最好是平滑过后的曲线，list or np.array
        startpoint:开始计算夹角的点，int，default:10
        begin_at_top:是否以曲线最高点为初始点，bool，default:True
    Output:
        angletrend:每个点与初始点之间割线的特征夹角构成的序列，list
        index:angletrend中每个点对应原始曲线的位置，list
    '''
    return angletrend, index
    
    
KRateTrend(seq, base_value):
    '''
    Input:
        To do:
    Output:
        To do:
    '''
    return
    

TypeofCurve(seq, tail_len=50):
    '''
    Input:
        seq:Fading data, better to be smoothed. list or np.array
        tail_len:判断结尾凹与结尾凸类型曲线时关注的结尾长度,default:50
    Output:
        Type:Type of curve, 0:纯下凹 1:纯上凸 2:结尾凹 3:结尾凸 4:直 -1:特殊类型. int
    '''
    return Type
    
@author: maliang
"""
from mypackage.curve_process import maxerror
import numpy as np
from numpy.linalg import norm
from math import acos
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from numpy import diff
import matplotlib.pyplot as plt
from mypackage.curve_process import smooth
from functools import reduce



def line(x, x0, x1, y0, y1):
    return y0 + (y1-y0)/(x1-x0)*(x-x0)
def CalAngle(seq):
    seq_sm = seq
    x = [i+1 for i in range (len(seq_sm))]
    seq_used = seq_sm
    x_used = x
    y = [line(x_0, x_used[0], x_used[-1], seq_used[0], seq_used[-1]) for x_0 in x_used]
    imax, vmax = maxerror(seq_used, y)
    if vmax <= 0:
        return 0
    NextPoint = imax
    while 1:
        NextPoint += 1
        if NextPoint == len(y)-1:
            break
        if seq_used[NextPoint] <= y[NextPoint]:
            break
    FirstPoint = np.array([x_used[0]/1000, seq_used[0]])
    UpPoint = np.array([x_used[imax]/1000, seq_used[imax]])
    EndPoint = np.array([x_used[NextPoint]/1000, seq_used[NextPoint]])
    DistA = norm(FirstPoint - EndPoint)
    DistB = norm(UpPoint - EndPoint)
    DistC = norm(FirstPoint - UpPoint)
    Cos_C = (DistA**2 + DistB**2 - DistC**2)/(2*DistA*DistB)
    Rad_C = acos(Cos_C)
    return Rad_C

def AngleTrend(seq, start_point=10, begin_at_top=True):
    angletrend = []
    usepart = seq
    usex = [i for i in range (len(seq))]  
    maxp = seq.index(max(seq))
    if begin_at_top:
        usepart = usepart[maxp:]
        usex = usex[maxp:]
    index = usex[start_point:]
    for i in range (start_point, len(usepart), 1):
        calpart = usepart[:i]
        angle = CalAngle(calpart)
        angletrend.append(angle)
    return angletrend, index

def TypeofCurve(seq, tail_len=50):
    x_used = [i for i in range (len(seq))]
    y = [line(x_0, x_used[0], x_used[-1], seq[0], seq[-1]) for x_0 in x_used]
    Type = -1
    lm = LinearRegression()
    lm.fit(np.array(x_used).reshape(-1,1), np.array(seq).reshape(-1,1))
    fitline = lm.predict(np.array(x_used).reshape(-1,1)).reshape(len(seq),)
    chebyshev = max(abs(fitline - np.array(seq)))
    mae = mean_absolute_error(np.array(seq), fitline)
    if chebyshev < 2e-3:
        Type = 4
        return Type
    if mae < 2e-3:
        Type = 4
        return Type
    if max(np.array(y) - np.array(seq)) < 1e-6:
        Type = 1
        return Type
    if max(np.array(seq) - np.array(y)) < 1e-6:
        Type = 0
        return Type
    if len(seq) > tail_len:
        tailjudge = []
        for i in range (tail_len-2):
            judgepart = seq[:-(i+3)]
            sub_x = [j for j in range (len(judgepart))]
            sub_y = [line(x_0, sub_x[0], sub_x[-1], judgepart[0], judgepart[-1]) for x_0 in sub_x]
            if max(np.array(sub_y) - np.array(judgepart)) < 1e-6:
                tailjudge.append(1)
            elif max(np.array(judgepart) - np.array(sub_y)) < 1e-6:
                tailjudge.append(0)
            else:
                tailjudge.append(-1)
        if tailjudge.count(1)/len(tailjudge) > 0.9:
            Type = 3
            return Type
        if tailjudge.count(0)/len(tailjudge) > 0.9:
            Type = 2
            return Type   
    return Type


def GetBase(seq, begin_at_top=True):
    length = int(0.08*len(seq))
    x = [i for i in range (len(seq))]
    diffs = diff(seq_sm)
    diffs_quarter = diffs[:int(len(seq)/4)]
    base_value = 0
    index = [i for i in range (len(seq))]
    if reduce(lambda x1, x2:x1*x2, diffs<=0) == True:
        base_center = np.where(diffs_quarter==diffs_quarter.max())[0][0]
        base_bottom = base_center - int(length/2)
        if base_bottom < 0:
            base_bottom = 0
        base_top = base_bottom + length
        index = x[base_bottom:base_top]
        base_value = np.mean(diffs[base_bottom:base_top])
    else:
        startp = int(0.02*len(seq))
        top = 0
        if begin_at_top == True:
            top = seq.index(max(seq))
        base = seq[top+startp:top+startp+length]
        index = x[top+startp:top+startp+length]
        diff_base = diff(base)
        base_value = np.mean(diff_base)
    return base_value, index

def KRateTrend(seq, base_value):
    diffs = np.diff(seq, n=1)
    k_rate_trend = diffs/base_value
    return k_rate_trend

def KKTrend(seq):
    return diff(seq, n=2)

def DiffWithScale(seq, scale=1):
    diffs = []
    for i in range (len(seq)-scale):
        diffs.append(seq[i+scale] - seq[i])
    index = [i for i in range (len(seq))]
    index = index[scale:]
    return diffs, index

def DiffDeviation(seq, base_value):
    diffs = np.diff(seq, n=1)
    diff_deviations = base_value - diffs
    return diff_deviations


    
names = np.load('p42names.npy')
seqs = np.load('p42data.npy')
seqs_sm = np.load('p42data_sm0.2.npy')

colors = [tuple([0.2,0.4,0.1]),\
          tuple([0.85,0.56,0.32]),\
          tuple([0.44,0.09,0.56]),\
          tuple([0.91,0.18,0.49]),\
          tuple([0.90,0.35,0.17])]
i = 6
for name, seq, seq_sm in zip(names, seqs, seqs_sm):
    if i >= 20:
        break
    
    angletrend, at_index = AngleTrend(seq_sm)
    base_value, base_index = GetBase(seq_sm)
    k_rate_trend = KRateTrend(seq_sm, base_value)
    kk_trend = KKTrend(seq_sm)
    kk_trend_sm = smooth(kk_trend, 0.06)
    diffs_deviation = DiffDeviation(seq_sm, base_value)
    
    fig = plt.figure(dpi = 300)
    left_axis = fig.add_subplot(111)
    plt.yticks(fontsize=4)
    right_axis = left_axis.twinx()
    plt.yticks(fontsize=4)
    ln0 = left_axis.plot(seq, label='original', color=colors[0], lw=0.6)
    ln1 = left_axis.plot(seq_sm, label='smooth', color=colors[0])
    left_axis.grid(True)
    left_axis.set_ylim(min(seq)*0.93, max(seq)*1.01)
    left_axis.set_ylabel('fading data')
    ln2 = right_axis.plot(at_index, angletrend, label='angle trend', color=colors[1])
    right_axis.axhline(0.05, linestyle='--', color='r', lw=0.8)
    right_axis.set_ylim(min(angletrend), 0.07)
    right_axis.set_ylabel('angle')
    plt.xlim(0, len(seq))
    lns = ln0+ln1+ln2
    labs = [l.get_label() for l in lns]
    left_axis.legend(lns, labs)
    plt.title('angle trend')
    plt.savefig('pkgtest/'+name+' angletrend'+'.jpg')
    plt.clf()
    plt.close(fig)
    
    fig = plt.figure(dpi = 300)
    left_axis = fig.add_subplot(111)
    plt.yticks(fontsize=4)
    right_axis = left_axis.twinx()
    plt.yticks(fontsize=4)
    ln0 = left_axis.plot(seq, label='original', color=colors[0], lw=0.6)
    ln1 = left_axis.plot(seq_sm, label='smooth', color=colors[0])
    left_axis.axvline(base_index[0], color='k', lw=0.8, linestyle='--')
    left_axis.axvline(base_index[-1], color='k', lw=0.8, linestyle='--')
    left_axis.grid(True)
    left_axis.set_ylim(min(seq)*0.93, max(seq)*1.01)
    left_axis.set_ylabel('fading data')
    ln2 = right_axis.plot(k_rate_trend, color=colors[2], label='k-rate')
    right_axis.axhline(1, color='k', lw=2.0)
    right_axis.set_ylim(min(k_rate_trend)*0.95, 4)
    right_axis.set_ylabel('k-rate')
    plt.xlim(0, len(seq))
    lns = ln0+ln1+ln2
    labs = [l.get_label() for l in lns]
    left_axis.legend(lns, labs)
    plt.title('k-rate')
    plt.savefig('pkgtest/'+name+' k-rate'+'.jpg')
    plt.clf()
    plt.close(fig)
    
    fig = plt.figure(dpi = 300)
    left_axis = fig.add_subplot(111)
    plt.yticks(fontsize=4)
    right_axis = left_axis.twinx()
    plt.yticks(fontsize=4)
    ln0 = left_axis.plot(seq, label='original', color=colors[0], lw=0.6)
    ln1 = left_axis.plot(seq_sm, label='smooth', color=colors[0])
    left_axis.grid(True)
    left_axis.set_ylim(min(seq)*0.93, max(seq)*1.01)
    left_axis.set_ylabel('fading data')
    ln2 = right_axis.plot(kk_trend, color=colors[3], label='2nd order diff(ori)', lw=0.3)
    ln3 = right_axis.plot(kk_trend_sm, color=colors[3], label='2nd order diff(smooth)')
    right_axis.axhline(0, color='k', lw=2.0)
    right_axis.set_ylim(-1e-6, 1e-7)
    right_axis.set_ylabel('2nd order diff')
    plt.xlim(0, len(seq))
    lns = ln0+ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    left_axis.legend(lns, labs)
    plt.title('2nd order difference')
    plt.savefig('pkgtest/'+name+' 2nd order diff.jpg')
    plt.clf()
    plt.close(fig)
    
    fig = plt.figure(dpi = 300)
    left_axis = fig.add_subplot(111)
    plt.yticks(fontsize=4)
    right_axis = left_axis.twinx()
    plt.yticks(fontsize=4)
    ln0 = left_axis.plot(seq, label='original', color=colors[0], lw=0.6)
    ln1 = left_axis.plot(seq_sm, label='smooth', color=colors[0])
    left_axis.axvline(base_index[0], color='k', lw=0.8, linestyle='--')
    left_axis.axvline(base_index[-1], color='k', lw=0.8, linestyle='--')
    left_axis.grid(True)
    left_axis.set_ylim(min(seq)*0.93, max(seq)*1.01)
    left_axis.set_ylabel('fading data')
    ln2 = right_axis.plot(diffs_deviation, color=colors[4], label='k-deviation')
    right_axis.axhline(0, color='k', lw=2.0)
    right_axis.set_ylim(-1e-5, 2e-4)
    plt.xlim(0, len(seq))
    lns = ln0+ln1+ln2
    labs = [l.get_label() for l in lns]
    left_axis.legend(lns, labs)
    plt.title('k-deviation')
    plt.savefig('pkgtest/'+name+' k-deviation.jpg')
    plt.clf()
    plt.close(fig)
    
    
    
    
    i += 1
 
    
    


#比率对于前期跳水敏感 可加入相对差值
#选基准应选取能反映曲线大趋势的阶段
#曲线形态种类
#化学反应速率/活性 对退化规律的影响

#！差分的尺度对近似微分效果的影响
#指标融合方式

#test DiffWithScale
#seq = seqs[27]
#seq_sm = smooth(seq, 0.2)
#fig = plt.figure(dpi = 700)
#left = fig.add_subplot(111)
#plt.yticks(fontsize=4)
#plt.grid(True)
#right = left.twinx()
#for i in range (50):
#    diff_value, index = DiffWithScale(seq_sm, i+1)
#    diff_value = np.array(diff_value)/(i+1)
#    left.plot(index, diff_value)
#right.plot(seq, lw=0.5)
#right.plot(seq_sm)
#plt.yticks(fontsize=4)
#plt.xlim(0, len(seq))
#plt.yticks(fontsize=4)
#plt.savefig('testdiff.jpg')
#plt.clf()
#plt.close(fig)
















