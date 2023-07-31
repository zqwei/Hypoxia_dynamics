import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.stats import spearmanr
import scipy.cluster.hierarchy as sch
import os

if __name__=='__main__':
    df = pd.read_csv('../data/datalist.csv', index_col=0)
    
    for n_row, row in df.iterrows():
        save_root = row['save_root']        
        if os.path.exists(save_root + 'FA_R3_ind.npz'):
            continue
        print(f'Processing {n_row} at {save_root}')
        
        dFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].astype('float16')
        A_center = np.load(save_root+'cell_center.npy')
        brain_map = np.load(save_root+'Y_ave.npy').astype('float')
        invalid_ = np.isnan(dFF_).sum(axis=1)>0
        invalid_ = invalid_ | (np.isinf(dFF_).sum(axis=1)>0)
        print(invalid_.mean())
        dFF_ = dFF_[~invalid_]
        A_center = A_center[~invalid_]
        zdFF_ = zscore(dFF_.astype('float'), axis=-1)
        del dFF_
        
        _ = np.load(save_root + 'FA_R2_ind.npz', allow_pickle=True)
        cluster_id = _['cluster_id']
        
        lind_, cnt_ = np.unique(cluster_id, return_counts=True)
        lind_ = lind_[np.argsort(-cnt_)]
        sub_cluster = []
        
        for c_id, ind_ in enumerate(lind_):
            print(c_id, ind_)
            idx_ = cluster_id==ind_
            zdFF_sub = zdFF_[idx_]
            num_cells = zdFF_sub.shape[0]
            if num_cells<200:
                # skip sub-clustering very small clusters
                sub_cluster.append(np.zeros(num_cells))
                continue
            step_ = num_cells//9000 + 1
            r_, p_ = spearmanr(zdFF_sub[::step_], axis=1)
            Z = sch.linkage(r_, method='ward')
            k = min(10, ((zdFF_sub[::step_].shape[0]//1000)+1)*2)
            print(f'number of sub-clustering {k}')
            labels = sch.fcluster(Z, k, criterion='maxclust')
            if step_>1:
                zCluster = np.array([zdFF_sub[::step_][labels==m].mean(axis=0) for m in range(1, k+1)])
                labels_sub = np.zeros(num_cells)
                n_splits = np.array_split(np.arange(num_cells).astype('int'), num_cells//1000+1)
                for n_cell in n_splits:
                    r_, p_ = spearmanr(zdFF_sub[n_cell], zCluster, axis=1)
                    labels_sub[n_cell] = np.argmax(r_[:len(n_cell), len(n_cell):], axis=1)
                sub_cluster.append(labels_sub)
            else:
                sub_cluster.append(labels-1)
                
        num_cells = zdFF_.shape[0]
        clusters_ind = np.zeros(num_cells).astype('int')-1
        subclusters_ind = np.zeros(num_cells).astype('int')-1
        
        subclusters_ind_max = 0
        for c_id, ind_ in enumerate(lind_):
            idx_ = np.where(cluster_id==ind_)[0]
            clusters_ind[idx_] = c_id
            subclusters_ind[idx_] = sub_cluster[c_id] + subclusters_ind_max
            subclusters_ind_max = subclusters_ind.max()+1
        subclusters_ind = subclusters_ind.astype('int16')
        clusters_ind = clusters_ind.astype('int16')
        
        np.savez(save_root + 'FA_R3_ind.npz', \
                 invalid_=invalid_, cluster_id_R2 = cluster_id, \
                 lind_=lind_, sub_cluster=np.array(sub_cluster, dtype='object'), \
                 clusters_ind=clusters_ind, subclusters_ind=subclusters_ind)