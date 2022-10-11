# -*- coding: utf-8 -*-
"""
Code for plotting the learning curves of saved runs

"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from file_saver_dumper_no_h5py import save_file, load_file, get_storage_path_reference
import json
import os

## Setup
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16 
M = 15 # moving avg parametner, odd number   

# Paths 
results_path = './results/delayedXOR_task/'
file_name = 'results'      


## Plot results
def movingmean(data_set, periods=3):
    data_set = np.array(data_set)
    if periods > 1:
        weights = np.ones(periods) / periods
        return np.convolve(data_set, weights, mode='valid') 
    else:
        return data_set
    
def iter_loss_acc(results_path, file_name, M, comment):
    all_f = os.listdir(results_path)
    flist = []
    for f in range(len(all_f)):
        if comment in all_f[f]:
            flist.append(all_f[f])
    
    if len(flist) > 0:
        plot_len = -1
        for f in range(len(flist)):
            file_path = results_path + flist[f]
            results_ = load_file(file_path,file_name,file_type='json')
            if f==0:   
                loss = np.expand_dims(movingmean(results_['loss_list'][0:plot_len] ,M),axis=0) 
            else:
                trial_loss = np.expand_dims(movingmean(results_['loss_list'][0:plot_len] ,M),axis=0)
                loss = np.concatenate((loss, trial_loss), axis=0) 
        
        # remove the worst run
        loss_auc = np.sum(loss, axis=1) 
        max_ind = np.argmax(loss_auc)
        loss = np.delete(loss, obj=max_ind, axis=0)        
        # remove the best run
        loss_auc = np.sum(loss, axis=1) 
        min_ind = np.argmin(loss_auc)
        loss = np.delete(loss, obj=min_ind, axis=0)
        
        mean_loss = np.mean(loss, axis=0)
        std_loss = np.std(loss, axis=0,ddof=1)
        iterlist = np.arange(M, M+loss.shape[1])
        
    else: # didn't fetch any files
        iterlist=np.empty(1000)
        iterlist[:]=np.nan
        mean_loss=np.empty(1000)
        mean_loss[:]=np.nan
        std_loss=np.empty(1000)
        std_loss[:]=np.nan
        
    return iterlist, mean_loss, std_loss 


comment_list = [['ModProp_Wab', 'c', (0.75, 0.9, 0.9),'ModProp_Wab'] ] 
sim_list = ['_lr0.0005', '_lr0.001'] 

samp_len = 50
fig0 = plt.figure()
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) 
for ii in range(len(comment_list)):
    bestMeas = np.Inf
    for sim_mode in sim_list:
        comm1_iterlist, mean1_comm, std1_comm = iter_loss_acc(results_path, file_name, M, comment_list[ii][0] + sim_mode)
        if np.mean(mean1_comm[-samp_len:]) < bestMeas: # take the best curve across different hyperparameters explored
            comm_iterlist, mean_comm, std_comm = (comm1_iterlist, mean1_comm, std1_comm)
            bestMeas = np.mean(mean1_comm[-samp_len:])
    plt.plot(comm_iterlist, mean_comm, color=comment_list[ii][1],label=comment_list[ii][3]) 
    plt.fill_between(comm_iterlist, mean_comm-std_comm, mean_comm+std_comm,color=comment_list[ii][2])
    
plt.legend();
plt.xlabel('Training Iterations (x20)')
plt.ylabel('Loss');
plt.title('Delayed XOR')
plt.ylim([0.0, 0.7])