import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
from sklearn.decomposition import FactorAnalysis
from tqdm import tqdm
import pickle
import os


def butter_lowpass_filter(data, cutoff, fs=1, order=5):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def f_FA(Y, L):
    from scipy.linalg import lstsq
    return lstsq(L, Y)[0]


def L_FA(Y, f):
    from scipy.linalg import lstsq
    return lstsq(f.T, Y.T)[0].T


if __name__=="__main__":
    df = pd.read_csv('../data/datalist.csv', index_col=0)
    
    for ind, row in df.iterrows():
        save_root = row['save_root']
        if os.path.exists(save_root+'FA_R1_ind.npz'):
            continue
        print(f'Processing {ind} at {save_root}')
        
        dFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].astype('float16')
        A_center = np.load(save_root+'cell_center.npy')
        brain_map = np.load(save_root+'Y_ave.npy').astype('float')
        # _ = np.load(save_root+'baseline_clusters.npz', allow_pickle=True)
        invalid_ = np.isnan(dFF_).sum(axis=1)>0
        invalid_ = invalid_ | (np.isinf(dFF_).sum(axis=1)>0)
        print(invalid_.mean())
        dFF_ = dFF_[~invalid_]
        A_center = A_center[~invalid_]
        zdFF_ = zscore(dFF_.astype('float'), axis=-1)
        del dFF_

        if not os.path.exists(save_root + 'FA_R1.pkl'):
            fa = FactorAnalysis(n_components=100, rotation='varimax')
            fa.fit(zdFF_.T)
            pickle.dump(fa, open(save_root + 'FA_R1.pkl', 'wb'))

        fa = pickle.load(open(save_root + 'FA_R1.pkl', 'rb'))
        lam = fa.components_.T
        loadings_ = lam.copy()
        f_dFF = f_FA(zdFF_, loadings_)
        filt_f = np.array([butter_lowpass_filter(_, cutoff=0.03, fs=1, order=5) for _ in f_dFF])
        noise_ratio_ = ((f_dFF-filt_f)**2).sum(axis=1)/(f_dFF**2).sum(axis=1)
        loadings_ = loadings_[:, noise_ratio_<0.3]
        # noise_ratio_ = noise_ratio_[noise_ratio_<0.3]

        loadings__ = loadings_.copy()
        loadings__[np.abs(loadings__)<0.2] = 0
        loadings__ = loadings_[:, (np.abs(loadings__)>0).sum(axis=0)>0]
        ind_ = np.argmax(np.abs(loadings__), axis=1)
        sign = np.array([loadings__[n, ind_[n]]>0 for n in range(len(ind_))]).astype('int')
        ind__ = ind_*(-1+2*sign)
        ind__[sign==0] = ind__[sign==0]-1

        np.savez(save_root + 'FA_R1_ind.npz', ind__=ind__, noise_ratio_=noise_ratio_)
    
    