import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import zscore
from scipy.stats import spearmanr
from tqdm import tqdm
sns.set(style='ticks', font_scale=1.)
df = pd.read_csv('../data/datalist.csv', index_col=0)


for ind, row in df.iterrows():
    save_root = row['save_root']
    if os.path.exists(save_root + 'baseline_stats.npz'):
        continue
    if not os.path.exists(save_root + '/cell_dff.npz'):
        continue
    print(ind)
    baseline_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['baseline'].astype('float16')
    _ = np.load(save_root+'baseline_clusters.npz', allow_pickle=True)
    invalid_ = _['invalid_']
    ev_thres = _['ev_thres']
    cell_cluster = _['cell_cluster']
    baseline_ = baseline_[~invalid_][ev_thres]
    time_stamp = np.load(save_root+'locs_cam.npy')/6000
    if len(time_stamp)<baseline_.shape[1]:
        baseline_ = baseline_[:, :len(time_stamp)]
    if len(time_stamp)>baseline_.shape[1]:
        time_stamp = time_stamp[:baseline_.shape[1]]
    num_cells = baseline_.shape[0]
    spilts = np.array_split(np.arange(num_cells).astype('int'), num_cells//1000)
    oxy_mean = np.load('../data/O2_internal.npz', allow_pickle=True)['oxy_mean']
    baseline_std = baseline_.astype('float').std(axis=1)
    baseline_mean = baseline_.astype('float').mean(axis=1)
    baseline_std_mean = baseline_std/baseline_mean
    idx_time = (time_stamp>10*60) & (time_stamp<60*60)
    oxy_interp = np.interp(time_stamp[idx_time], np.arange(oxy_mean.shape[0])+1, oxy_mean)
    r_cell = []
    p_cell = []
    
    for split_ in tqdm(spilts):
        r_, p_ = spearmanr(baseline_[split_][:, idx_time], oxy_interp[None, :], axis=1)
        r_cell.append(r_[-1][:-1])
        p_cell.append(p_[-1][:-1])
    r_cell = np.concatenate(r_cell)
    p_cell = np.concatenate(p_cell)
    idx_hypo = (time_stamp>35*60) & (time_stamp<40*60)
    idx_norm = (time_stamp>15*60) & (time_stamp<20*60)
    hypo_baseline_mean = baseline_[:, idx_hypo].astype('float').mean(axis=1)
    norm_baseline_mean = baseline_[:, idx_norm].astype('float').mean(axis=1)
    hypo_baseline_std = baseline_[:, idx_hypo].astype('float').std(axis=1)
    norm_baseline_std = baseline_[:, idx_norm].astype('float').std(axis=1)

    np.savez(save_root + 'baseline_stats.npz', \
             baseline_std=baseline_std, \
             baseline_mean=baseline_mean, \
             r_cell=r_cell, p_cell=p_cell, \
             hypo_baseline_mean=hypo_baseline_mean, \
             norm_baseline_mean=norm_baseline_mean, \
             hypo_baseline_std=hypo_baseline_std, \
             norm_baseline_std=norm_baseline_std)