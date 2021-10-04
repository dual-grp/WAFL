import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
plt.rcParams.update({'font.size': 14})

def average_smooth(data, window_len=20, window='flat'):
    results = []
    if window_len<3:
        return data
    for i in range(len(data)):
        x = data[i]
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('numpy.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='valid')
        results.append(y[window_len-1:])
    return np.array(results)

alpha = ["0.05","0.1","0.5","1.0","5.0"]
glob_acc = []
test_loss = []
robust_pgd_loss = []
robust_pgd_acc = []
for i in range(len(alpha)):
    str_files = "Cifar10_FedRob_0.05_0.4_"+alpha[i] + "_20u_64b_2_10_avg"
    hf = h5py.File("./Result_cifa/"+'{}.h5'.format(str_files), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_test_loss = np.array(hf.get('rs_test_loss')[:])
    
    #rs_target_acc = np.array(hf.get('rs_target_acc')[:])
    #rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    #rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    
    rs_robust_pgd_loss = np.array(hf.get('rs_robust_pgd_loss')[:])
    rs_robust_pgd_acc = np.array(hf.get('rs_robust_pgd_acc')[:])
    
    glob_acc.append(average_smooth(rs_glob_acc))
    test_loss.append(average_smooth(rs_test_loss))
    robust_pgd_loss.append(average_smooth(rs_robust_pgd_loss))
    robust_pgd_acc.append(average_smooth(rs_robust_pgd_acc))