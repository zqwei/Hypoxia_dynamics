import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr, zscore
from tqdm import tqdm
sns.set(style='ticks', font_scale=1.)
df = pd.read_csv('../data/datalist_gfap_gc6f.csv', index_col=0)


ind = 6
row = df.iloc[ind]

save_root = row['save_root']
baseline_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['baseline'].astype('float16')
# dFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].astype('float16')
_ = np.load(save_root+'baseline_clusters.npz', allow_pickle=True)
invalid_ = _['invalid_']
ev_thres = _['ev_thres']
cell_cluster = _['cell_cluster']
baseline_ = baseline_[~invalid_][ev_thres]
# dFF_ = dFF_[~invalid_][ev_thres]
time_stamp = np.load(save_root+'locs_cam.npy')/6000
num_cells = baseline_.shape[0]
spilts = np.array_split(np.arange(num_cells).astype('int'), num_cells//1000)
oxy_mean = np.load('../data/O2_internal.npz', allow_pickle=True)['oxy_mean']
_ = np.load(save_root + 'baseline_stats.npz', allow_pickle=True)
baseline_std = _['baseline_std']
baseline_mean = _['baseline_mean']
r_cell = _['r_cell']
p_cell = _['p_cell']
hypo_baseline_mean = _['hypo_baseline_mean']
norm_baseline_mean = _['norm_baseline_mean']
hypo_baseline_std = _['hypo_baseline_std']
norm_baseline_std = _['norm_baseline_std']
idx_ = (baseline_std>2) & (baseline_std/baseline_mean>0.05)
baseline_change = hypo_baseline_mean/norm_baseline_mean
baseline_std_mean = baseline_std/baseline_mean
A_center = np.load(save_root+'cell_center.npy')[~invalid_][ev_thres]
brain_map = np.load(save_root+'Y_ave.npy')


baseline_mean_thres = 20
idx_ = baseline_mean<baseline_mean_thres
z_loc = np.arange(brain_map.shape[0])
fig, ax = plt.subplots(2, 5, figsize=(60, 15))
ax = ax.flatten()
for n in range(10):
    z_idx = (z_loc>=n*3) & (z_loc<(n+1)*3) # change to 6 for neural data
    ax[n].imshow(brain_map[z_idx].max(0), cmap = plt.cm.gray)
    z_idx = (A_center[~idx_, 0]>=n*3) & (A_center[~idx_, 0]<(n+1)*3)
    ax[n].scatter(A_center[~idx_, 1][z_idx], A_center[~idx_, 2][z_idx], s=1, c=A_center[~idx_, 0][z_idx])
    ax[n].set_axis_off()
plt.tight_layout()
plt.show()


idx_F = (baseline_mean>baseline_mean_thres) & (baseline_std>2) & (baseline_std_mean>0.05) & (p_cell<0.001)
idx_time = (time_stamp>10*60) & (time_stamp<60*60)
oxy_interp = np.interp(time_stamp[idx_time], np.arange(oxy_mean.shape[0])+1, oxy_mean)
zbaseline_ = zscore(baseline_[idx_F][:, idx_time].astype('float'), axis=-1)


# r_cell[idx_F].max(), r_cell[idx_F].min()
r_idx = np.arange(-0.8, 0.9, 0.1)
r_idx = np.r_[-1, r_idx, 1]
len_ = len(r_idx)
ref = []
for n in range(len_-1):
    idx__ = (r_cell[idx_F]>r_idx[n]) & (r_cell[idx_F]<r_idx[n+1])
    ref.append(zbaseline_[idx__].mean(axis=0))
ref = np.array(ref)

num_cells = idx_F.sum()
spilts_ref = np.array_split(np.arange(num_cells).astype('int'), num_cells//1000)
ref_cluster = ref.shape[0]
r_cell_ = []
p_cell_ = []

for split_ in tqdm(spilts_ref):
    r_, p_ = spearmanr(zbaseline_[split_], ref, axis=1)
    r_cell_.append(r_[:-ref_cluster][:, -ref_cluster:])
    p_cell_.append(p_[:-ref_cluster][:, -ref_cluster:])

r_cell_ = np.concatenate(r_cell_)
r_cell_max = r_cell_.max(axis=1)
r_cell_max_idx = np.argmax(r_cell_, axis=1)
np.savez(save_root+'O2_clusters.npz', idx_F=idx_F, ref=ref, r_cell_=r_cell_)

colors = plt.cm.rainbow(np.linspace(0, 1, ref_cluster))
for i in range(ref_cluster):
    plt.plot(ref[i], color=colors[i])
plt.show()

A_center_ = A_center[idx_F]
r_cell_max_idx_ = r_cell_max_idx/r_cell_max_idx.max()

fig, ax = plt.subplots(2, 5, figsize=(60, 15))
ax = ax.flatten()
for n in range(10):
    z_idx = (z_loc>=n*3) & (z_loc<(n+1)*3)
    ax[n].imshow(brain_map[z_idx].max(0), cmap = plt.cm.gray)
    z_idx = (A_center_[:, 0]>=n*3) & (A_center_[:, 0]<(n+1)*3) & (r_cell_max>0.7)
    ax[n].scatter(A_center_[z_idx, 1], A_center_[z_idx, 2], s=1, c=r_cell_max_idx_[z_idx], vmax=1, vmin=0, cmap=plt.cm.rainbow)
    ax[n].set_axis_off()
plt.tight_layout()
plt.show()