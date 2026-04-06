import pandas as pd
import numpy as np
import os
import shutil
df = pd.read_csv('../../data/datalist_huc_ablation.csv', index_col=0)

for ind, row in df.iterrows():
    ephys_ = row['dir_'] + '/proc_data/'
    save_root = row['save_root']
    shutil.copyfile(ephys_+'cell_center_affine_registered.npy', save_root+'/cell_center_affine_registered.npy')