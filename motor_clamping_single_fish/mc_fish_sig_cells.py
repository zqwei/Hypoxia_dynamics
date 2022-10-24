from utils import *

def motor_clamp_cells(idx):
    row = df.iloc[idx]
    save_root = row['save_root']
    locs_cam = np.load(save_root + 'locs_cam.npy')
    swim_start_idx = File(save_root + 'data.mat', 'r')['data']['swimStartIndT2'][()].squeeze()
    valid_F = np.load(save_root + 'motor_cells.npz', allow_pickle=True)['valid_F']
    vdFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'][valid_F]
    vdFF_ = vdFF_.astype('float16')
    idx1_ = np.load(save_root+'/motor_clamp_trials.npz', allow_pickle=True)['idx1_']
    idx2_ = np.load(save_root+'/motor_clamp_trials.npz', allow_pickle=True)['idx2_']

    # removing baseline significant cells
    F_cond1 = []
    idx_ = idx1_.copy()
    num_sample = len(idx_)
    for n in range(num_sample):
        swim_time_start = swim_start_idx[idx_[n]]
        F_idx = (locs_cam<=swim_time_start).sum()
        if F_idx==0:
            continue
        F_cond1_ = vdFF_[:, F_idx+1:F_idx+3].mean(axis=1)-vdFF_[:, F_idx-1:F_idx+1].mean(axis=1)
        F_cond1.append(F_cond1_)
    F_cond1 = np.array(F_cond1)

    F_cond2 = []
    idx_ = idx2_.copy()
    num_sample = len(idx_)
    for n in range(num_sample):
        swim_time_start = swim_start_idx[idx_[n]]
        F_idx = (locs_cam<=swim_time_start).sum()
        if F_idx==0:
            continue
        F_cond2_ = vdFF_[:, F_idx+1:F_idx+3].mean(axis=1)-vdFF_[:, F_idx-1:F_idx+1].mean(axis=1)
        F_cond2.append(F_cond2_)
    F_cond2 = np.array(F_cond2)

    num_cells = vdFF_.shape[0]
    sig_cells = np.zeros(num_cells)
    for n_ in tqdm(range(num_cells)):
        _, p = mannwhitneyu(F_cond1[:, n_], F_cond2[:, n_])
        sig_cells[n_] = p


    # baseline significant cells
    F_cond1 = []
    idx_ = idx1_.copy()
    num_sample = len(idx_)
    for n in range(num_sample):
        swim_time_start = swim_start_idx[idx_[n]]
        F_idx = (locs_cam<=swim_time_start).sum()
        if F_idx==0:
            continue
        F_cond1_ = vdFF_[:, F_idx-1:F_idx+3].mean(axis=1)
        F_cond1.append(F_cond1_)
    F_cond1 = np.array(F_cond1)

    F_cond2 = []
    idx_ = idx2_.copy()
    num_sample = len(idx_)
    for n in range(num_sample):
        swim_time_start = swim_start_idx[idx_[n]]
        F_idx = (locs_cam<=swim_time_start).sum()
        if F_idx==0:
            continue
        F_cond2_ = vdFF_[:, F_idx-1:F_idx+3].mean(axis=1)
        F_cond2.append(F_cond2_)
    F_cond2 = np.array(F_cond2)

    num_cells = vdFF_.shape[0]
    sig_cells_baseline = np.zeros(num_cells)
    for n_ in tqdm(range(num_cells)):
        _, p = mannwhitneyu(F_cond1[:, n_], F_cond2[:, n_])
        sig_cells_baseline[n_] = p

    np.savez(save_root+'/motor_clamp_sig_cells.npz', \
             valid_F=valid_F, sig_cells=sig_cells, \
             sig_cells_baseline=sig_cells_baseline)

for n in range(3, 9):
    motor_clamp_cells(n)