import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
import pandas as pd
sns.set(style='ticks', font_scale=1.)
df = pd.read_csv('../data/datalist.csv', index_col=0)
valid_list = [df.iloc[n]['registration_root']!='None' for n in range(len(df))]
df = df[valid_list]
atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas.npy')

idx_neg_list = []
idx_pos_list = []
num_animal = 0
for ind, row in df.iterrows():
    save_root = row['save_root']
    reg_root = row['registration_root']
    A_center = np.load(save_root+'cell_center_affine_registered.npy')
    idx_neg = np.load(reg_root+'/proc_data/idx_neg.npy')
    idx_pos = np.load(reg_root+'/proc_data/idx_pos.npy')
    idx_neg_list.append(np.hstack([A_center[idx_neg], np.ones((idx_neg.sum(),1))*num_animal]))
    idx_pos_list.append(np.hstack([A_center[idx_pos], np.ones((idx_pos.sum(),1))*num_animal]))
    num_animal = num_animal+1
idx_neg_list = np.vstack(idx_neg_list)
idx_pos_list = np.vstack(idx_pos_list)

rz, ry, rx = 2.5, 5, 5
result_ = np.zeros(atlas.shape)
n_list = idx_pos_list # idx_neg_list
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
# np.savez(brain_map_folder + 'brain_map_idx_neg.npz', result_=result_.astype('uint8'))
np.savez(brain_map_folder + 'brain_map_idx_pos.npz', result_=result_.astype('uint8'))