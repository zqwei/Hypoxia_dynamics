from utils import *

def motor_clamp_trials(idx, thres1, thres2):
    row = df.iloc[idx]
    save_root = row['save_root']
    ephys_data = File(save_root + 'data.mat', 'r')['data']
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
    swim_intv_thres = 1.0
    
    u_bout_power, l_bout_power, u_swim_bout_, l_swim_bout_, \
    u_swim_bout, l_swim_bout = thres1

    idx = (swim_start>150) & (swim_start<1200)
    idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
    idx = idx & (swim_bout_length>u_swim_bout) & (swim_bout_length<l_swim_bout)
    idx_ = np.where(idx)[0]
    idx1 = idx_ 
    idx = (swim_start>1500) & (swim_start<2400)
    idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
    idx = idx & (swim_bout_length>u_swim_bout_) & (swim_bout_length<l_swim_bout_)
    idx_ = np.where(idx)[0]
    idx2 = idx_

    u_bout_power, l_bout_power = thres2
    idx = (swim_start>150) & (swim_start<1200)
    idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
    idx_ = np.where(idx)[0]
    idx1_ = idx_ 
    idx = (swim_start>1500) & (swim_start<2400)
    idx = idx & (swim_bout_power>u_bout_power) & (swim_bout_power<l_bout_power) & (swim_intv>swim_intv_thres)
    idx_ = np.where(idx)[0]
    idx2_ = idx_

    np.savez(save_root+'/motor_clamp_trials.npz', \
             idx1=idx1, idx2=idx2, \
             idx1_=idx1_, idx2_=idx2_)


### Fish 3
'''
### fish01 
8dpf huc:h2b-gc7f CID 7694
- ch1 ch2 0.1-1000Hz; both are good, ch2 may be better
- the fish was recorded about two hours after paralysis
- The fish is hypoxic, obvious swim EEG in the beginning of ZTS1

ZTS1-oxygen-spon, 2 hours
- continuous open-loop visual stimulation, vel 0.5
- gc7f imging at 2.4Hz, 61 planes/stack, 300um, 5um step
- 0-1200sec: 16 mg/L high O2 water; many swims
- 1200-2400sec: low O2 water; gradually fewer swims; more rest EEG
- 2400-4800sec: 16 mg/L high O2 water; many swims
- 4800-6000sec: low O2 water; gradually fewer swims; more rest EEG
- 2400-4800sec: 16 mg/L high O2 water; no swims and EEG in the beginning of water switch; and more swims later
'''
idx = 3
u_bout_power = 0.05
l_bout_power = 0.09
u_swim_bout_ = 0.13
l_swim_bout_ = 0.18
u_swim_bout = 0.13
l_swim_bout = 0.18
thres1 = (u_bout_power, l_bout_power, u_swim_bout_, \
          l_swim_bout_, u_swim_bout, l_swim_bout)
u_bout_power = 0.05
l_bout_power = 0.09
thres2 = (u_bout_power, l_bout_power)
motor_clamp_trials(idx, thres1, thres2)


### Fish 4
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
u_bout_power = 0.65
l_bout_power = 0.8
u_swim_bout_ = 0.2
l_swim_bout_ = 0.3
u_swim_bout = 0
l_swim_bout = 1
thres1 = (u_bout_power, l_bout_power, u_swim_bout_, \
          l_swim_bout_, u_swim_bout, l_swim_bout)
u_bout_power = 0.4
l_bout_power = 0.8
thres2 = (u_bout_power, l_bout_power)
motor_clamp_trials(idx, thres1, thres2)


### Fish 5
'''
fish01 6dpf huc:h2b-gc7f; vglut2a-DsRed CID 7878
- ch1 300-1000Hz, ch2 0.1-1000Hz; ch2 EEG is quite small, no obvious EEG; 
- Use opto software

ZTS1-oxygen-CL, 1 hour
- close-loop visual stimulation, gain 0.01, vel 1
- gc7f imging at 2.4Hz, 61 planes/stack, 300um, 5um step
- 0-1200sec: normal water; many swims 
- 1200-2400sec: low O2 water; fewer, but still some swims
- 2400-3600sec: normal water; many swims with small amplitude
'''
idx = 5
u_bout_power = 0.12
l_bout_power = 0.16
u_swim_bout_ = 0.2
l_swim_bout_ = 0.3
u_swim_bout = 0.2
l_swim_bout = 0.3
thres1 = (u_bout_power, l_bout_power, u_swim_bout_, \
          l_swim_bout_, u_swim_bout, l_swim_bout)
u_bout_power = 0.1
l_bout_power = 0.2
thres2 = (u_bout_power, l_bout_power)
motor_clamp_trials(idx, thres1, thres2)


### Fish 6
'''
fish01 6dpf huc:h2b-gc7f CID 7708
- ch1 ch2 0.1-1000Hz; both are good, ch1 may be better
- the fish was recorded about 1 hour after paralysis
- The fish is hypoxic, obvious EEG in the beginning

ZTS1-oxygen-spon, 1 hour
- continuous open-loop visual stimulation, vel 0.5
- gc7f imging at 2Hz, 71 planes/stack, 350um, 5um step
- 0-1200sec: 16 mg/L high O2 water; many swims
- 1200-2400sec: low O2 water; gradually fewer swims; more rest EEG
- 2400-3600sec: 16 mg/L high O2 water; many swims
'''
idx = 6
u_bout_power = 0.6
l_bout_power = 0.8
u_swim_bout_ = 0.1
l_swim_bout_ = 0.3
u_swim_bout = 0.1
l_swim_bout = 0.3
thres1 = (u_bout_power, l_bout_power, u_swim_bout_, \
          l_swim_bout_, u_swim_bout, l_swim_bout)
u_bout_power = 0.5
l_bout_power = 1.1
thres2 = (u_bout_power, l_bout_power)
motor_clamp_trials(idx, thres1, thres2)


### Fish 7
'''
fish02 6dpf huc:h2b-gc7f; vglut2a-DsRed CID 7878
- ch1 300-1000Hz, ch2 0.1-1000Hz; ch2 EEG is small, but obvious; 
- Use opto software

ZTS2-oxygen-OL, 1 hour
- open-loop visual stimulation, gain 0, vel 1
- gc7f imging at 2.4Hz, 61 planes/stack, 300um, 5um step
- 0-1200sec: normal water; many swims 
- 1200-2400sec: low O2 water; still some swims
- 2400-3600sec: normal water; many swims with small amplitude
'''
idx = 7
u_bout_power = 0.15
l_bout_power = 0.25
u_swim_bout_ = 0.15
l_swim_bout_ = 0.25
u_swim_bout = 0.15
l_swim_bout = 0.25
thres1 = (u_bout_power, l_bout_power, u_swim_bout_, \
          l_swim_bout_, u_swim_bout, l_swim_bout)
u_bout_power = 0.15
l_bout_power = 0.3
thres2 = (u_bout_power, l_bout_power)
motor_clamp_trials(idx, thres1, thres2)


### Fish 8
'''
fish02 6dpf huc:h2b-gc7f CID 7708
- ch1 ch2 0.1-1000Hz; both are good, ch1 may be better
- The fish is hypoxic, few swim in the beginning

ZTS1-oxygen-spon, 2 hours
- continuous open-loop visual stimulation, vel 0.5
- gc7f imging at 2.4Hz, 61 planes/stack, 350um, 5um step
- 0-1200sec: 16 mg/L high O2 water; many swims
- 1200-2400sec: low O2 water; gradually fewer swims; more rest EEG
- 2400-4800sec: 16 mg/L high O2 water; many swims
- 4800-6000sec: low O2 water; gradually fewer swims; more rest EEG
- 6000-7200sec: 16 mg/L high O2 water; many swims
'''
idx = 8
u_bout_power = 0.5
l_bout_power = 0.6
u_swim_bout_ = 0.15
l_swim_bout_ = 0.3
u_swim_bout = 0.15
l_swim_bout = 0.3
thres1 = (u_bout_power, l_bout_power, u_swim_bout_, \
          l_swim_bout_, u_swim_bout, l_swim_bout)
u_bout_power = 0.4
l_bout_power = 0.8
thres2 = (u_bout_power, l_bout_power)
motor_clamp_trials(idx, thres1, thres2)

