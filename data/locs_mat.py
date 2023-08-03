import pandas as pd
import numpy as np
from h5py import File
import os
import shutil
df = pd.read_csv('datalist.csv', index_col=0)

for ind, row in df.iterrows():
    ephys_ = row['dir_'] + '/ephys/analysis/'
    save_root = row['save_root']
    if os.path.exists(save_root + 'locs_cam.npy'):
        continue
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    try:
        shutil.copyfile(ephys_+'data.mat', save_root+'/data.mat')
    except:
        print(f'Missing data file at {save_root}')
        pass
    shutil.copyfile(ephys_+'x3.mat', save_root+'/x3.mat')
    # shutil.copyfile(ephys_+'locs_cam.mat', save_root+'/locs_cam.mat')
    
    x3 = File(ephys_ + 'x3.mat')['x3'][()].squeeze()
    locs_cam = np.where((x3[:-1]<3.8) & (x3[1:]>3.8))[0]+1
    # shape_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].shape[1]
    # print(locs_cam.shape, shape_)
    np.save(save_root + 'locs_cam.npy', locs_cam.astype('int'))