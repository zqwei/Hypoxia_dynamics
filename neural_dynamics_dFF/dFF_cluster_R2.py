import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import FactorAnalysis
import pickle
import os
from dFF_cluster_R1 import butter_lowpass_filter, f_FA, L_FA


def loading2labels(loadings_, min_cluster=100, min_w=0.2, thres_large_cluster=0.9):
    loadings_pos = loadings_.copy()
    loadings_neg = loadings_.copy()
    loadings_pos[loadings_<0] = 0
    loadings_neg[loadings_>0] = 0
    loadings__ = np.concatenate([loadings_pos, loadings_neg], axis=1)
    # remove low weight components
    loadings__[np.abs(loadings__)<min_w] = 0
    loadings__ = loadings__[:, (np.abs(loadings__)>0).sum(axis=0)>min_cluster]

    for _ in range(3):
        ind_ = np.argmax(np.abs(loadings__), axis=1)
        lind_, cnt_ = np.unique(ind_, return_counts=True)
        valid_ind_ = np.zeros(loadings__.shape[1]).astype('bool')
        for m_, l_ in enumerate(lind_):
            if cnt_[m_]>min_cluster:
                valid_ind_[l_] = True
        loadings__ = loadings__[:, valid_ind_]
        # print(valid_ind_.sum())
    ind_ = np.argmax(np.abs(loadings__), axis=1)
    lind_, cnt_ = np.unique(ind_, return_counts=True)
    loadings__ = loadings__[:, np.argsort(-cnt_)]
    ind_ = np.argmax(np.abs(loadings__), axis=1)
    if (ind_==0).mean()>thres_large_cluster:
        ind_max = ind_.max()
        valid_sec_ind = ((np.abs(loadings__[:, 1:])>0).sum(axis=1)>0) & (ind_==0)
        if valid_sec_ind.mean()>0.5:
            ind_[valid_sec_ind] = np.argmax(np.abs(loadings__[valid_sec_ind, 1:]), axis=1)+ind_max
    return ind_


if __name__=='__main__':
    df = pd.read_csv('../data/datalist.csv', index_col=0)
    max_cluster_size = 0.05
    
    for n_row, row in df.iterrows():
        save_root = row['save_root']
        if os.path.exists(save_root + 'FA_R2_ind.npz'):
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
        
        num_cells = zdFF_.shape[0]
        fa = pickle.load(open(save_root + 'FA_R1.pkl', 'rb'))
        _ = np.load(save_root + 'FA_R1_ind.npz', allow_pickle=True)
        noise_ratio_ = _['noise_ratio_']

        loadings_ = fa.components_.T
        loadings_ = loadings_[:, noise_ratio_<0.3]

        ind_ = loading2labels(loadings_, min_cluster=100, min_w=0.07, thres_large_cluster=0.9)
        cluster_id = ind_.copy()

        lind_, cnt_ = np.unique(ind_, return_counts=True)
        max_cluster_size = 0.05
        max_cluster = len(cnt_)//2+5
        valid_cluster = cnt_/num_cells<max_cluster_size
        run_loop = (~valid_cluster).sum()>0

        res_nFA = []
        res_n_noise_ratio_ = []
        res_nLayer = []
        res_nFA_ind = []   
        
        max_cluster_size = 0.05
        thres_large_cluster=0.9
        cnt_pre = cnt_.copy()
        nLayer = 0

        while run_loop:
            nLayer = nLayer+1
            print(f'Run nlayer {nLayer} for {(~valid_cluster).sum()} large clusters.......')
            print(f'size of large clusters {cnt_[~valid_cluster]}')
            for m_ind in lind_[~valid_cluster]:
                idx_ = cluster_id==m_ind
                num_cluster = min(max_cluster, idx_.sum()//1000)
                print(f'clustering for {m_ind} using {num_cluster} clusters')
                fa_sub = FactorAnalysis(n_components=num_cluster, rotation='varimax')
                zdFF_sub = zdFF_[idx_]
                fa_sub.fit(zdFF_sub.T)
                loadings_ = fa_sub.components_.T
                print(f'cleaning up clusters for {m_ind}')
                f_dFF = f_FA(zdFF_sub, loadings_)
                filt_f = np.array([butter_lowpass_filter(_, cutoff=0.03, fs=1, order=5) for _ in f_dFF])
                noise_ratio_ = ((f_dFF-filt_f)**2).sum(axis=1)/(f_dFF**2).sum(axis=1)
                res_nFA.append(fa_sub)
                res_n_noise_ratio_.append(noise_ratio_)
                res_nLayer.append(nLayer)
                res_nFA_ind.append(m_ind)
            n_indx = np.where(np.array(res_nLayer)==nLayer)[0]

            lind_max = cluster_id.max()
            lind_max_layer_ = cluster_id.max()
            for m_ in n_indx:
                idx_ = cluster_id==res_nFA_ind[m_]
                loadings_ = res_nFA[m_].components_.T
                noise_ratio_ = res_n_noise_ratio_[m_]    
                loadings_ = loadings_[:, noise_ratio_<0.3]
                ind_sub = loading2labels(loadings_, min_cluster=100, min_w=0.07, thres_large_cluster=0.9)
                if (ind_sub>0).sum()>0:
                    indx_sub = np.where(idx_)[0]
                    cluster_id[indx_sub[ind_sub>0]] = ind_sub[ind_sub>0]+lind_max
                    lind_max = cluster_id.max()
            lind_, cnt_ = np.unique(cluster_id, return_counts=True)
            max_cluster = max((lind_max-lind_max_layer_)//len(n_indx)+5, 20)
            valid_cluster = cnt_/num_cells<max_cluster_size
            non_factored_cluster = np.where(cnt_[:len(cnt_pre)]/cnt_pre>0.9)[0]
            valid_cluster[non_factored_cluster] = True
            print(np.where(~valid_cluster)[0])
            run_loop = (~valid_cluster).sum()>0
            cnt_pre = cnt_.copy()
            
        np.savez(save_root + 'FA_R2_ind.npz', \
                 cluster_id=cluster_id, \
                 res_nFA=np.array(res_nFA, dtype='object'), \
                 res_n_noise_ratio_=np.array(res_n_noise_ratio_, dtype='object'), \
                 res_nLayer=np.array(res_nLayer), res_nFA_ind=np.array(res_nFA_ind))
        
        

