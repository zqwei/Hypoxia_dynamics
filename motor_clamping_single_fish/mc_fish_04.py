from utils import *

'''
### fish02 
6dpf huc:h2b-gc7f CID 7285
- ch1 ch2 0.1-1000Hz; both are good, ch1 is better
- still many swims, obvious swim EEG in the beginning

ZTS1-oxygen-spon, 1 hour
- continuous open-loop visual stimulation, vel 0.3; 30-sec interval
- gc7f imging at 2Hz, 61 planes/stack, 300um, 5um step
- 0-1200sec: normal water; many swims
- 1200-2400sec: low O2 water; gradually fewer swims; some rest EEG
- 2400-3600sec: normal water; many swims
'''

idx = 4
row = df.iloc[idx]
# ephys_ = row['dir_'] + '/ephys/analysis/'
save_root = row['save_root']

ephys_data = File(save_root + 'data.mat', 'r')['data']
locs_cam = File(save_root + 'locs_cam.mat', 'r')['locs_cam'][()].squeeze()

fltCh1 = ephys_data['fltCh2'][()].squeeze()
back1 = ephys_data['back2'][()].squeeze()
swim_ = (fltCh1-back1)*10000
swim_[swim_<0]=0

swim_start_idx = ephys_data['swimStartIndT2'][()].squeeze()
swim_end_idx = ephys_data['swimEndIndT2'][()].squeeze()
swim_start = swim_start_idx/6000
swim_end = swim_end_idx/6000
swim_bout_power = ephys_data['swimPower2'][()].squeeze()

swim_intv = np.diff(swim_start)
swim_intv_before = np.r_[swim_intv[0], swim_intv]
swim_intv = np.r_[swim_intv, swim_intv[-1]]

swim_bout_length = swim_end - swim_start

# # plot(locs_cam/6000, F)
# time_ = np.arange(len(swim_))/6000
# plt.plot(time_, swim_)
# plt.xlabel('Time (s)')
# plt.ylabel('Swim power')
# sns.despine()
# plt.show()

### Swim power
idx = swim_start<1200
x, y = ecdf(np.log(swim_bout_power[idx]))
plt.plot(np.exp(x), y, '-k')

idx = (swim_start>1500) & (swim_start<2400)
x, y = ecdf(np.log(swim_bout_power[idx]))
plt.plot(np.exp(x), y, '-r')

plt.xlabel('swim power')
plt.ylabel('CDF')
# plt.xlim([0, 2])
sns.despine()
plt.show()

### Swim length
idx = swim_start<1200
x, y = ecdf(np.log(swim_bout_length[idx]))
plt.plot(np.exp(x), y, '-k')

idx = (swim_start>1500) & (swim_start<2400)
x, y = ecdf(np.log(swim_bout_length[idx]))
plt.plot(np.exp(x), y, '-r')

plt.xlabel('swim bout length (sec)')
plt.ylabel('CDF')
# plt.xlim([0, 2])
sns.despine()
plt.show()

### Stats
u_bout_power = 0.65
l_bout_power = 0.8
u_swim_bout_ = 0.2
l_swim_bout_ = 0.3
u_swim_bout = 0
l_swim_bout = 1

idx = swim_start<1200
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power)
idx = idx & (swim_bout_length>u_swim_bout) & (swim_bout_length<l_swim_bout)
print('inter-swim-interval')
print(np.mean(swim_intv[idx]))


idx = (swim_start>1500) & (swim_start<2400)
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power)
idx = idx & (swim_bout_length>u_swim_bout_) & (swim_bout_length<l_swim_bout_)
print('inter-swim-interval')
print(np.mean(swim_intv[idx]))

swim_intv_thres = 1.0

idx = swim_start<1200
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
idx = idx & (swim_bout_length>u_swim_bout) & (swim_bout_length<l_swim_bout)
print('average swim power')
print(np.mean(swim_bout_power[idx]))
print('# trials')
print(np.sum(idx))

idx = (swim_start>1500) & (swim_start<2400)
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
idx = idx & (swim_bout_length>u_swim_bout_) & (swim_bout_length<l_swim_bout_)
print('average swim power')
print(np.mean(swim_bout_power[idx]))
print('# trials')
print(np.sum(idx))

### Clamped trials
u_bout_power = 0.4
l_bout_power = 0.8

idx = swim_start<1200
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
print('average swim power')
print(np.mean(swim_bout_power[idx]))
print('# trials')
print(np.sum(idx))

idx = (swim_start>1500) & (swim_start<2400)
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
print('average swim power')
print(np.mean(swim_bout_power[idx]))
print('# trials')
print(np.sum(idx))

idx = swim_start<1200
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
# idx = idx & (swim_bout_length>u_swim_bout) & (swim_bout_length<l_swim_bout)
idx_ = np.where(idx)[0]
swim_start_idx = swim_start_idx.astype('int')

num_sample = len(idx_)
xq = np.arange(-600,4800)/6000;
xq_ = len(xq)
dat_ = np.zeros((num_sample, xq_))

for n in range(num_sample):
    swim_time_start = swim_start_idx[idx_[n]]-600
    dat_[n] = swim_[swim_time_start:swim_time_start+xq_]

mean_ = np.mean(dat_, axis=0);
sem_ = np.std(dat_, axis=0)/np.sqrt(num_sample);
plt.plot(xq, mean_, '-k', lw=2);
plt.plot(xq, mean_+sem_, '-k', lw=1)
plt.plot(xq, mean_-sem_, '-k', lw=1)
# plt.plot(xq, dat_.T, '-k', lw=1)


idx = (swim_start>1500) & (swim_start<2400)
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
# idx = idx & (swim_bout_length>u_swim_bout_) & (swim_bout_length<l_swim_bout_)
idx_ = np.where(idx)[0]
swim_start_idx = swim_start_idx.astype('int')

num_sample = len(idx_)
xq = np.arange(-600,4800)/6000;
xq_ = len(xq)
dat_ = np.zeros((num_sample, xq_))

for n in range(num_sample):
    swim_time_start = swim_start_idx[idx_[n]]-600
    dat_[n] = swim_[swim_time_start:swim_time_start+xq_]

mean_ = np.mean(dat_, axis=0);
sem_ = np.std(dat_, axis=0)/np.sqrt(num_sample);
plt.plot(xq, mean_, '-r', lw=2);
plt.plot(xq, mean_+sem_, '-r', lw=1)
plt.plot(xq, mean_-sem_, '-r', lw=1)
# plt.plot(xq, dat_.T, '-r', lw=1)

plt.xlabel('Time (s)')
plt.ylabel('Swim power')
plt.xlim([-0.1, 0.8])
sns.despine()
plt.show()

### neural data
dFF_ = np.load(save_root + 'cell_dff.npz', allow_pickle=True)['dFF'].astype('float')
A_center = np.load(save_root+'cell_center.npy')
brain_map = np.load(save_root+'Y_ave.npy').astype('float')

num_cells = dFF_.shape[0]
valid_F = np.ones(num_cells).astype('bool')
for n_ in range(num_cells):
    if np.isnan(dFF_[n_]).sum()>0:
        valid_F[n_] = False
vdFF_ = dFF_[valid_F]
dFF_ = None

F_cond1 = []
idx = swim_start<1200
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
idx_ = np.where(idx)[0]
num_sample = len(idx_)

for n in range(num_sample):
    swim_time_start = swim_start_idx[idx_[n]]
    F_idx = (locs_cam<=swim_time_start).sum()
    if F_idx==0:
        continue
    F_cond1_ = vdFF_[:, F_idx+1:F_idx+3].mean(axis=1) - vdFF_[:, F_idx-3:F_idx-1].mean(axis=1)
    F_cond1.append(F_cond1_)
F_cond1 = np.array(F_cond1)
    
F_cond2 = []
idx = (swim_start>1500) & (swim_start<2400)
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
idx_ = np.where(idx)[0]
num_sample = len(idx_)

for n in range(num_sample):
    swim_time_start = swim_start_idx[idx_[n]]
    F_idx = (locs_cam<=swim_time_start).sum()
    if F_idx==0:
        continue
    F_cond2_ = vdFF_[:, F_idx+1:F_idx+3].mean(axis=1) - vdFF_[:, F_idx-3:F_idx-1].mean(axis=1)
    F_cond2.append(F_cond2_)
F_cond2 = np.array(F_cond2)

num_cells = vdFF_.shape[0]
sig_cells = np.zeros(num_cells)

for n_ in tqdm(range(num_cells)):
    _, p = mannwhitneyu(F_cond1[:, n_], F_cond2[:, n_])
    sig_cells[n_] = p
    
### Save data
u_bout_power = 0.65
l_bout_power = 0.8
u_swim_bout_ = 0.2
l_swim_bout_ = 0.3
u_swim_bout = 0
l_swim_bout = 1

idx = swim_start<1200
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
idx = idx & (swim_bout_length>u_swim_bout) & (swim_bout_length<l_swim_bout)
idx_ = np.where(idx)[0]
idx1 = idx_ 

idx = (swim_start>1500) & (swim_start<2400)
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
idx = idx & (swim_bout_length>u_swim_bout_) & (swim_bout_length<l_swim_bout_)
idx_ = np.where(idx)[0]
idx2 = idx_

u_bout_power = 0.4
l_bout_power = 0.8

idx = swim_start<1200
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
idx_ = np.where(idx)[0]
idx1_ = idx_ 

idx = (swim_start>1500) & (swim_start<2400)
idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
idx_ = np.where(idx)[0]
idx2_ = idx_

np.savez(save_root+'/motor_clamp_sig_cells.npz', \
         valid_F=valid_F, sig_cells=sig_cells, \
         idx1=idx1, idx2=idx2, \
         idx1_=idx1_, idx2_=idx2_)


