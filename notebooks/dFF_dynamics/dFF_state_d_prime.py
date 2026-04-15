import numpy as np
import pandas as pd
import os
from scipy.stats import wilcoxon, mannwhitneyu
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../data/datalist.csv', index_col=0)


def mannwhitneyu_p(x, y):
    return mannwhitneyu(x.astype('float'), y.astype('float'), axis=1)[1]


def mannwhitneyu_p_unpack(x):
    return mannwhitneyu_p(*x)


for ind, row in df[10:3:-1].iterrows():
    save_root = row['save_root']
    print(ind, save_root)
    if os.path.exists(save_root + 'hypoxia_d_prime.npz'):
        continue
    _ = np.load(save_root + 'FA_R3_ind.npz', allow_pickle=True)
    invalid_ = _['invalid_']
    dFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].astype('float16')[~invalid_]
    A_center = np.load(save_root+'cell_center_affine_registered.npy')[~invalid_]
    
    _ = np.load(save_root+'LR_HMM_state_v2.npz', allow_pickle=True)
    most_likely_states = _['most_likely_states']
    n_state = _['n_state']
    h_state = _['h_state']
    
    nstate = h_state
    prob_1 = most_likely_states==nstate
    nstate = n_state
    try:
        prob_2 = most_likely_states==nstate
    except:
        prob_2 = most_likely_states==nstate[0]
    
    num_cells = dFF_.shape[0]
    splits = np.array_split(np.arange(num_cells).astype('int'), num_cells//1000)
    pools = [[dFF_[s][:, prob_1], dFF_[s][:, prob_2]] for s in splits]
    with Pool(cpu_count()-3) as p:
        p_value = p.map(mannwhitneyu_p_unpack, pools)
    dFF_p_value = np.concatenate(p_value)
    
    dFF_d_prime = (dFF_[:, prob_1].astype('float').mean(axis=1) - dFF_[:, prob_2].astype('float').mean(axis=1))
    dFF_d_prime = dFF_d_prime/np.sqrt((dFF_[:, prob_1].astype('float').var(axis=1) + dFF_[:, prob_2].astype('float').var(axis=1))/2)

    np.savez(save_root + 'hypoxia_d_prime.npz', dFF_p_value=dFF_p_value, dFF_d_prime=dFF_d_prime, A_center=A_center)