from utils import *

for ind, row in df.iterrows():
    save_root = row['save_root']
    _ = np.load(save_root + '/baseline_oxy.npz', allow_pickle=True)
    valid_F = _['valid_F']
    r_ = _['r_']
    p_ = _['p_']
    mean_baseline_ = _['mean_baseline_']
    baseline_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['baseline'].astype('float16')[valid_F]
    neg_baseline_ = baseline_[r_<-0.1].mean(axis=0)
    pos_baseline_ = baseline_[r_>0.8].mean(axis=0)
    
    np.savez(save_root+'/baseline_oxy_ave.npz', \
             neg_baseline_=neg_baseline_, pos_baseline_=pos_baseline_)