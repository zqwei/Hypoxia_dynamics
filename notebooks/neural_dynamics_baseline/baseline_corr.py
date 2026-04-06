from utils import *

for ind, row in df.iterrows():
    save_root = row['save_root']
    if os.path.exists(save_root + '/baseline_oxy.npz'):
        continue
    if not os.path.exists(save_root + '/cell_dff.npz'):
        continue
    dFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].astype('float16')
    baseline_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['baseline'].astype('float16')
    num_cells = dFF_.shape[0]
    valid_F = np.ones(num_cells).astype('bool')
    for n_ in range(num_cells):
        if np.isnan(dFF_[n_]).sum()>0:
            valid_F[n_] = False
        if dFF_[n_].max()>10:
            valid_F[n_] = False
    baseline_ = baseline_[valid_F]
    mean_baseline_ = baseline_.mean(axis=0)
    
    num_cells = baseline_.shape[0]
    p_ = np.zeros(num_cells)
    r_ = np.zeros(num_cells)
    
    for n in tqdm(range(num_cells)):
        r, p = spearmanr(mean_baseline_, baseline_[n])
        p_[n] = p
        r_[n] = r
    
    np.savez(save_root + '/baseline_oxy.npz', \
             valid_F=valid_F, mean_baseline_=mean_baseline_, \
             p_ = p_, r_ = r_)