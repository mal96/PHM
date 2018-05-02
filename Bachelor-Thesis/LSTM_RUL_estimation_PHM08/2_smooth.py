# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:41:35 2018

对正规化后的传感器监测数据平滑

@author: maliang
"""
import os

import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt

def smooth(seq, frac):
    lowess = sm.nonparametric.lowess
    x = list(range(len(seq)))
    seq_smoothed = lowess(seq, x, frac=frac, return_sorted=False)
    return seq_smoothed

engine_list = [str(i + 1) for i in range(218)]  # All engines
sensor_list = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
               's13', 's14', 's15', 's17', 's20', 's21']   # All sensitive sensors


# creative folders to save smooth figures
figure_path = 'figures/smooth/'
for sensor_name in sensor_list:
    if not os.path.exists(figure_path+sensor_name+'/'):
        os.mkdir(figure_path+sensor_name+'/')
        
        
data_path = 'data_with_engine_ID/'
for engine_ID in engine_list:
    df_original = pd.read_csv(data_path+engine_ID+'_transformed.csv', index_col=0)
    df_smooth = pd.DataFrame()
    for icol in range (df_original.shape[1]):
        col_name = df_original.columns[icol]
        if col_name in sensor_list:
            # smooth, save
            seq_raw = np.array(df_original[col_name])
            seq_smoothed = smooth(seq_raw, frac=0.05)
            
            # plot to observe, could be commented
            plt.plot(seq_raw, label='original')
            plt.plot(seq_smoothed, label='smooth')
            plt.legend()
            plt.title('Engine #'+engine_ID+', '+col_name)
            plt.savefig(figure_path+col_name+'/'+engine_ID+'.jpg', dpi=200)
            plt.clf()
            plt.close()
            
            # add to df_smooth
            df_smooth[col_name] = seq_smoothed
        else:
            # directly save
            df_smooth[col_name] = df_original[col_name]
    df_smooth.to_csv(data_path+engine_ID+'_smooth.csv')
    
    
    
            
    




