import pandas as pd
import shutil
df = pd.read_csv('datalist.csv', index_col=0)

for ind, row in df.iterrows():
    ephys_ = row['dir_'] + '/ephys/analysis/'
    save_root = row['save_root']
    shutil.copyfile(ephys_+'data.mat', save_root+'/data.mat')
    shutil.copyfile(ephys_+'locs_cam.mat', save_root+'/locs_cam.mat')
