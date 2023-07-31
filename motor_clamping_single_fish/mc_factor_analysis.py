from utils import *
from scipy.stats import zscore
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import Rotator

def mc_factors(idx):
    row = df.iloc[idx]
    save_root = row['save_root']
    _ = np.load(save_root+'/motor_clamp_sig_cells.npz', allow_pickle=True)
    valid_F = _['valid_F']
    sig_cells = _['sig_cells']
    sig_cells_baseline = _['sig_cells_baseline']
    sig_cell_idx = (sig_cells<0.01) # |(sig_cells_baseline<0.01)
    print(idx, sig_cell_idx.mean())
    sig_F = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'][valid_F][sig_cell_idx]
    dat_ = zscore(sig_F[200:].astype('float'), axis=1)
    sig_F = None

    FA_=FactorAnalysis(n_components=40, svd_method='randomized', random_state=None, iterated_power=3, rotation='varimax')
    FA_.fit(dat_.T)
    score_ = FA_.noise_variance_
    score_ = 1 - score_
    s_thres = 0.1
    s_idx_ = score_>s_thres
    print(s_idx_.sum(), s_idx_.mean())

    lam=FA_.components_.T
    Rot_=Rotator(power=4)
    loadings, rotation_mtx, phi=Rot_._promax(lam)

    n_factor = loadings.shape[1]
    sparse_loadings = loadings.copy()
    for n in range(n_factor):
        lam_ = sparse_loadings[:, n]
        sparse_loadings[np.abs(lam_)<0.1, n] = 0

    invalid_factor_neurons = (np.abs(sparse_loadings)>0).sum(axis=1)==0
    sparse_loadings = sparse_loadings[~invalid_factor_neurons]
    num_neurons = (np.abs(sparse_loadings)>0.4).sum(axis=0)
    isort_ = np.argsort(-num_neurons)
    sparse_loadings = sparse_loadings[:, isort_]

    np.savez(save_root + 'mc_factors.npz', \
             lam=lam, sparse_loadings=sparse_loadings, \
             invalid_factor_neurons=invalid_factor_neurons)

for n in range(3, 9):
    mc_factors(n)

