import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
import pandas as pd
import os
df = pd.read_csv('../data/datalist.csv', index_col=0)
valid_list = [df.iloc[n]['registration_root']!='None' for n in range(len(df))]
atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas.npy')

idx_motor_list = []
num_animal = 0
for ind, row in df.iterrows():
    save_root = row['save_root']
    if not os.path.exists(save_root+'/motor_clamp_sig_cells.npz'):
        continue
    reg_root = row['registration_root']
    A_center = np.load(save_root+'cell_center_affine_registered.npy')
    _ = np.load(save_root+'/motor_clamp_sig_cells.npz', allow_pickle=True)
    valid_F = _['valid_F']
    sig_cells = _['sig_cells']
    idx_motor = sig_cells<0.05 #0.01
    idx_motor_list.append(np.hstack([A_center[valid_F][idx_motor], np.ones((idx_motor.sum(),1))*num_animal]))
    num_animal = num_animal+1
idx_motor_list = np.vstack(idx_motor_list)


rz, ry, rx = 2.5, 5, 5
result_ = np.zeros(atlas.shape)
n_list = idx_motor_list 
ind_loc = (n_list[:, 2]<atlas.shape[1]-1) & (n_list[:, 2]>0)
ind_loc = ind_loc & (n_list[:, 0]<atlas.shape[0]-1) & (n_list[:, 0]>0)
ind_loc = ind_loc & (n_list[:, 1]<atlas.shape[2]-1) & (n_list[:, 1]>0)
n_list = n_list[ind_loc]
z_, x_, y_ = np.round(n_list[:, :3].T).astype('int')

for n_animal in range(num_animal):
    ind = (n_list[:, 3]==n_animal) 
    result_anm = np.zeros(atlas.shape)
    result_anm[z_[ind], y_[ind], x_[ind]]=1
    result_anm = gaussian_filter(result_anm, [rz, ry, rx], truncate=1.0)
    result_ = result_ + (result_anm>result_anm.max()*.1).astype('int')

brain_map_folder = '/nrs/ahrens/Ziqiang/Motor_clamp/Brain_maps/'
np.savez(brain_map_folder + 'brain_map_motor_clamp_0_5.npz', result_=result_.astype('uint8'))
# np.savez(brain_map_folder + 'brain_map_motor_clamp_0_1.npz', result_=result_.astype('uint8'))