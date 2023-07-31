import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
from sklearn.decomposition import FactorAnalysis
from tqdm import tqdm
df = pd.read_csv('../data/datalist.csv', index_col=0)
fa = FactorAnalysis(n_components=10, rotation='varimax')


for ind, row in df.iterrows():
    if ind<17:
        continue
    save_root = row['save_root']
    baseline_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['baseline'].astype('float16')
    A_center = np.load(save_root+'cell_center.npy')
    brain_map = np.load(save_root+'Y_ave.npy').astype('float')
    _ = np.load(save_root+'baseline_clusters.npz', allow_pickle=True)
    invalid_ = _['invalid_']
    ev_thres = _['ev_thres']
    cell_cluster = _['cell_cluster']
    # cluster_act_mat = _['cluster_act_mat']
    baseline_ = baseline_[~invalid_][ev_thres]
    A_center = A_center[~invalid_][ev_thres]
    zbaseline_ = zscore(baseline_.astype('float'), axis=-1)
    del baseline_
    num_cluster = cell_cluster.max().astype('int')+1
    cell_subcluster = np.zeros(zbaseline_.shape[0])-1
    
    num_subcluster = 0
    for n_c in tqdm(range(num_cluster)):
        idx_ = cell_cluster==n_c
        num_cells = idx_.sum()
        tmp_dat = zbaseline_[idx_]
        tmp_subcluster = np.zeros(num_cells)
        fa.fit(tmp_dat.T)

        cluster_mat = []
        for n_cluster in range(10):
            idx = fa.components_[n_cluster]<-0.1
            if idx.sum()>100:
                cluster_mat.append(zbaseline_[cell_cluster==n_c][idx].mean(axis=0))
            idx = fa.components_[n_cluster]>0.1
            if idx.sum()>100:
                cluster_mat.append(zbaseline_[cell_cluster==n_c][idx].mean(axis=0))

        max_corr = np.ones(num_cells)
        subcluster = np.zeros(num_cells) - 1
        n_splits = np.array_split(np.arange(num_cells).astype('int'), num_cells//1000)
        cluster_mat = np.array(cluster_mat)
        len_cluster = cluster_mat.shape[0]
        for n_cell in n_splits:
            r, p = spearmanr(tmp_dat[n_cell], cluster_mat, axis=1)
            r = r[:-len_cluster, -len_cluster:]
            idx_r = np.argmax(r, axis=1)
            max_corr[n_cell] = np.max(r, axis=1)
            subcluster[n_cell] = idx_r
        cell_cluster_ = subcluster.copy()
        if (max_corr<0.4).sum()>0:
            cell_cluster_[max_corr<0.4] = -1
            cell_cluster_ = cell_cluster_+1
        cell_subcluster[idx_] = cell_cluster_+num_subcluster
        num_subcluster = cell_cluster_.max()+1+num_subcluster
    label_, cnt_ = np.unique(cell_subcluster, return_counts=True)
    cell_subcluster_mat = []
    num_subcluster = cell_subcluster.max().astype('int')+1
    for n in range(num_subcluster):
        cell_subcluster_mat.append(zbaseline_[cell_subcluster==n].mean(axis=0))
    cell_subcluster_mat = np.array(cell_subcluster_mat)
    
    locs_cam = np.load(save_root+'locs_cam.npy')
    oxy_ = np.load('../data/O2_internal.npz', allow_pickle=True)['oxy_mean']
    # oxygen data interpolation

    f = interp1d(np.arange(3600), oxy_)
    time_ = locs_cam/6000
    time_=time_-time_[0]
    oxy_new = f(time_[time_<3600-1])
    time__ = time_[time_<3600-1]
    num_cells = cell_subcluster_mat.shape[0]
    p_ = np.zeros(num_cells)
    r_ = np.zeros(num_cells)

    for n in tqdm(range(num_cells)):
        r, p = spearmanr(oxy_new[time__>300], cell_subcluster_mat[n,:oxy_new.shape[0]][time__>300]) # correlation between 5-60min
        p_[n] = p
        r_[n] = r
    
    np.savez(save_root+'baseline_subclusters.npz', \
             cell_subcluster=cell_subcluster, \
             cell_subcluster_mat=cell_subcluster_mat, \
             p_oxy = p_, r_oxy = r_)