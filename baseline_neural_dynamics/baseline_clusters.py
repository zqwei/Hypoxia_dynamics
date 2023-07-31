import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
df = pd.read_csv('../data/datalist.csv', index_col=0)
pca_ = PCA(n_components=20)

for ind, row in df.iterrows():
    save_root = row['save_root']
    if os.path.exists(save_root+'baseline_clusters.npz'):
        continue
    print(f'Processing {ind} at {save_root}')
    dFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].astype('float16')
    baseline_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['baseline'].astype('float16')
    A_center = np.load(save_root+'cell_center.npy')
    brain_map = np.load(save_root+'Y_ave.npy').astype('float')
    invalid_ = np.isnan(dFF_).sum(axis=-1)>0
    baseline_ = baseline_[~invalid_][:, :] ## remove the first 600 frames
    A_center = A_center[~invalid_]
    del dFF_
    zbaseline_ = zscore(baseline_.astype('float'), axis=-1)
    del baseline_

    pca_.fit(zbaseline_)
    zbaseline_pca = pca_.transform(zbaseline_)
    zbaseline_res = pca_.inverse_transform(zbaseline_pca)
    exp_var = 1- ((zbaseline_res-zbaseline_)**2).sum(axis=-1)/(zbaseline_**2).sum(axis=-1)
    ev_thres = exp_var>0.9
    zbaseline_res = zbaseline_res[ev_thres].astype('float16')
    zbaseline_pca = zbaseline_pca[ev_thres].astype('float16')
    del zbaseline_
    z_pca_max = np.abs(zbaseline_pca).max(axis=-1, keepdims=True)
    ind_pca_max = np.argmax(np.abs(zbaseline_pca), axis=-1)
    sign_pca_max  = np.array([zbaseline_pca[n, i]>0 for n, i in enumerate(ind_pca_max)])
    sparse_pca_ = (np.abs(zbaseline_pca)>z_pca_max*.4).sum(axis=-1)
    cluster_act_mat = []
    cluster_label = []
    num_cells = zbaseline_res.shape[0]
    cell_cluster = np.zeros(num_cells).astype('int8') - 1
    n_cluster = -1
    for n_c in range(20):
        idx_ = (ind_pca_max==n_c) & (sparse_pca_==1) & ~sign_pca_max
        if idx_.sum()>100:
            n_cluster = n_cluster+1
            cluster_act_mat.append(zbaseline_res[idx_].mean(axis=0))
            cluster_label.append(n_cluster)
            cell_cluster[idx_] = n_cluster
        idx_ = (ind_pca_max==n_c) & (sparse_pca_==1) & sign_pca_max
        if idx_.sum()>100:
            n_cluster = n_cluster+1
            cluster_act_mat.append(zbaseline_res[idx_].mean(axis=0))
            cluster_label.append(n_cluster)
            cell_cluster[idx_] = n_cluster
    cluster_act_mat = np.array(cluster_act_mat)
    cell_cluster = np.array(cell_cluster)
    cluster_label = np.array(cluster_label)
    max_corr = np.ones(num_cells)
    n_splits = np.array_split(np.arange(num_cells).astype('int'), num_cells//1000)
    len_cluster = cluster_act_mat.shape[0]
    for n_cell in tqdm(n_splits):
        r, p = spearmanr(zbaseline_res[n_cell], cluster_act_mat, axis=1)
        r = r[:-len_cluster, -len_cluster:]
        idx_r = np.argmax(r, axis=1)
        max_corr[n_cell] = np.max(r, axis=1)
        cell_cluster[n_cell] = idx_r
    cell_cluster_ = cell_cluster.copy()
    cell_cluster_[max_corr<0.4] = -1
    cluster_act_mat_ = []
    for n_c in cluster_label:
        cluster_act_mat_.append(zbaseline_res[cell_cluster_==n_c].mean(axis=0))
    cluster_act_mat_ = np.array(cluster_act_mat_)
    np.savez(save_root+'baseline_clusters.npz', \
             invalid_=invalid_, ev_thres=ev_thres, \
             cell_cluster=cell_cluster_, cluster_act_mat=cluster_act_mat_)