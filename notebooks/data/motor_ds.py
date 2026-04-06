import pandas as pd
import numpy as np
from h5py import File
import os
# df = pd.read_csv('datalist.csv', index_col=0)
# df = pd.read_csv('datalist_huc_ablation.csv', index_col=0)
# df = pd.read_csv('datalist_th1_gc6f.csv', index_col=0)
df = pd.read_csv('../../data/datalist_gfap_gc6f_v2.csv', index_col=0)

for ind, row in df.iterrows():
    if ind>10:
        continue
    ephys_ = row['dir_'] + '/ephys/analysis/'
    save_root = row['save_root']
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if os.path.exists(save_root + 'swim_ds.npy'):
        print(save_root + 'swim_ds.npy')
        continue
    if not os.path.exists(save_root + 'data.mat'):
        print(f'Missing data file at {save_root}')
        continue
    locs_cam = np.load(save_root + 'locs_cam.npy')
    len_cam = np.unique(np.diff(locs_cam)).min()
    ephys_data = File(save_root + 'data.mat', 'r')['data']
    fltCh1 = ephys_data['fltCh2'][()].squeeze()
    back1 = ephys_data['back2'][()].squeeze()
    swim_ = (fltCh1-back1)
    swim_[swim_<0]=0
    
    swim_ds = np.zeros(len(locs_cam))
    for n_ in range(len_cam):
        swim_ds = swim_ds + swim_[locs_cam-n_]
    
    np.save(save_root + 'swim_ds.npy', swim_ds)