import numpy as np
import pandas as pd
from scipy.stats import zscore
import os
df = pd.read_csv('../data/datalist.csv', index_col=0)

for n_row, row in df.iterrows():
    save_root = row['save_root']
    if os.path.exists(save_root + 'FA_R3_dynamics.npz'):
        continue
    print(f'Processing {n_row} at {save_root}')
    dFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].astype('float16')
    _ = np.load(save_root + 'FA_R3_ind.npz', allow_pickle=True)
    invalid_ = _['invalid_']
    clusters_ind = _['clusters_ind'] # large clusters
    subclusters_ind = _['subclusters_ind'] # sub-clusters
    dFF_ = dFF_[~invalid_]
    cluster_act = []
    num_cells = []
    num_cluster = subclusters_ind.max().astype('int')+1
    for n in range(num_cluster):
        idx_ = subclusters_ind==n
        if idx_.sum()<50:
            continue
        zdFF_sub = zscore(dFF_[idx_].astype('float'), axis=1)
        cluster_act.append(zdFF_sub.mean(axis=0))
        num_cells.append(idx_.sum())
    cluster_act = np.array(cluster_act)
    num_cells = np.array(num_cells)
    np.savez(save_root + 'FA_R3_dynamics.npz', \
             cluster_act=cluster_act, num_cells = num_cells)

