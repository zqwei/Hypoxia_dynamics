from utils import *
from scipy.stats import spearmanr

for ind, row in df.iterrows():
    save_root = row['save_root']
    ephys_ = row['dir_'] + '/ephys/analysis/'
    dFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].astype('float16')
    baseline_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['baseline'].astype('float16')
    A_center = np.load(save_root+'cell_center.npy')
    brain_map = np.load(save_root+'Y_ave.npy').astype('float')
    locs_cam = File(ephys_ + 'locs_cam.mat', 'r')['locs_cam'][()].squeeze()
    num_cells = dFF_.shape[0]
    valid_F = np.ones(num_cells).astype('bool')
    for n_ in range(num_cells):
        if np.isnan(dFF_[n_]).sum()>0:
            valid_F[n_] = False
    # z_ = brain_map.shape[0]
    # valid_F = valid_F & (A_center[:, 0]<(z_-1.5)) & (A_center[:, 0]>1.5)
    baseline_ = baseline_[valid_F]
    mean_baseline_ = baseline_.mean(axis=0)
    
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(oxy_, '-k', lw=0.5)
    # len_ = min(len(locs_cam), len(mean_baseline_))
    # ax2.plot((locs_cam/6000)[:len_], mean_baseline_[:len_], '-r', lw=2)
    # ax1.vlines([1200, 2400], ymin=0, ymax=1.0, colors='k', linestyles='--')
    # ax1.set_xlim([0, 3600])
    # ax1.set_ylabel('Normalized oxygen level')
    # ax2.set_ylabel('Baseline F',color="r")
    # ax1.set_xlabel('Time (s)')
    # # sns.despine()
    # plt.show()
    
    num_cells = baseline_.shape[0]
    p_ = np.zeros(num_cells)
    r_ = np.zeros(num_cells)
    
    for n in tqdm(range(num_cells)):
        r, p = spearmanr(mean_baseline_, baseline_[n])
        p_[n] = p
        r_[n] = r
    
    np.savez(save_root + '/baseline_oxy.npz', \
             valid_F=valid_F, mean_baseline_=mean_baseline_, \
             p_ = p_, r_ = r_, locs_cam = locs_cam)