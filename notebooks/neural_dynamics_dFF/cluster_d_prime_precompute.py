from utils_cluster_anm import *
# df = pd.read_csv('../data/datalist_huc_h2b_gc7f.csv', index_col=0)
df = pd.read_csv('../data/datalist_gfap_gc6f.csv', index_col=0)

# for ind, row in df[:-1].iterrows():\
for ind, row in df.iterrows():
    save_root = row['save_root']
    time_stamp = np.load(save_root+'locs_cam.npy')/6000
    _ = np.load(save_root+'baseline_clusters.npz', allow_pickle=True)
    invalid_ = _['invalid_']
    ev_thres = _['ev_thres']
    _ = np.load(save_root+'O2_clusters.npz', allow_pickle=True)
    idx_F = _['idx_F']
    dFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].astype('float16')[~invalid_][ev_thres][idx_F]
    print(ind, dFF_.shape)
    if ind==8:
        dFF_ = dFF_[:, :-1]

    prob_2 = ((time_stamp/60<20) & (time_stamp/60>10))
    prob_1 = ((time_stamp/60<40) & (time_stamp/60>25))
    x = dFF_[:, prob_1].astype('float')
    y = dFF_[:, prob_2].astype('float')
    normalized = np.sqrt((np.nanvar(x, axis=1) + np.nanvar(y, axis=1))/2)

    prob_2 = ((time_stamp/60<20) & (time_stamp/60>10))
    time1 = 25
    time2 = 35
    prob_1 = ((time_stamp/60<time2) & (time_stamp/60>time1))
    x = dFF_[:, prob_1].astype('float') # hypoxia
    y = dFF_[:, prob_2].astype('float') # normxia
    d_prime_vec = (y.mean(axis=1) - x.mean(axis=1))/normalized
    np.save(save_root+'/cell_d_prime_for_cluster.npy', d_prime_vec)