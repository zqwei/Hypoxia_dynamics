import numpy as np
from scipy.signal import detrend, find_peaks, hilbert
from scipy.ndimage import gaussian_filter1d


def butter_bandpass(bands, fs, order=5):
    from scipy.signal import butter
    return butter(order, bands, fs=fs, btype='bandpass', analog=False, output='sos')


def butter_bandpass_filter(data, bands, fs, order=5):
    from scipy.signal import sosfiltfilt
    sos = butter_bandpass(bands, fs, order=order)
    y = sosfiltfilt(sos, data, padtype=None)
    return y


def butter_lowpass(cutoff, fs, order=5):
    from scipy.signal import butter
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_cfilter(data, cutoff, fs, order=5):
    from scipy.signal import lfilter
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=5):
    from scipy.signal import filtfilt
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def moving_average(data, span):
    import pandas as pd
    return pd.Series(data).rolling(window=span, min_periods=1, center=True).mean().values


def moving_std(data, span):
    import pandas as pd
    return pd.Series(data).rolling(window=span, min_periods=1, center=True).std().values


def moving_perc(data, span, perc=10):
    import pandas as pd
    return pd.Series(data).rolling(window=span, min_periods=1, center=True).quantile(perc/100).values


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def nan_fills(y):
    nans, x= nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y


def ave_remove_outlier(x, p=[5, 95]):
    low_, high_ = np.percentile(x, p)
    return x[(x>low_)&(x<high_)].mean()


def swim_detection_from_fltswim(fltCh1, thre = 2.5, d_=6000):
    aa1 = np.diff(fltCh1);
    peaksT1 = (aa1[0:-1] > 0) * (aa1[1:] < 0);
    peaksIndT1 = np.argwhere(peaksT1>0).squeeze();
    x_ = np.arange(0,0.10001,0.00001);
    th1 = np.zeros(fltCh1.size,);
    back1 = np.zeros(fltCh1.size,);
    
    last_i=0
    
    for i in np.arange(0,fltCh1.size-d_,d_):
        peaksIndT1_ = np.argwhere(peaksT1[0:(i+d_)]>0).squeeze();
        a1,_ = np.histogram(fltCh1[peaksIndT1_], x_)
        a1=a1.astype('f4')
        mx1 = (np.argwhere(a1 == a1.max())).min()
        mn1_ind=np.argwhere(a1[0:mx1] < (a1[mx1]/200))
        if (mn1_ind.size>0):
            mn1=mn1_ind.max()
        else:
            mn1=0;
        th1[i:(i+d_+1)] = x_[mx1] + thre*(x_[mx1]-x_[mn1]);
        back1[i:(i+d_+1)] = x_[mx1] ;
        last_i=i
    
    th1[(last_i+d_+1):] = th1[last_i+d_];
    back1[(last_i+d_+1):] = back1[last_i+d_] ;
    
    burstIndT1 = peaksIndT1[np.argwhere((fltCh1-th1)[peaksIndT1]>0).squeeze()];
    burstT1=np.zeros(fltCh1.size);
    burstT1[burstIndT1]=1;
    burstBothT = np.zeros(fltCh1.size);
    burstBothT[burstIndT1] = 1;
    burstBothIndT = np.argwhere(burstBothT>0).squeeze();
    interSwims = np.diff(burstBothIndT);
    swimEndIndB = np.argwhere(interSwims > 600).squeeze();
    swimEndIndB = np.append(swimEndIndB,burstBothIndT.size-1)
    
    swimStartIndB=0;
    swimStartIndB = np.append(swimStartIndB,swimEndIndB[:-1]+1);
    nonSuperShort = np.argwhere(swimEndIndB != swimStartIndB).squeeze();
    
    swimEndIndB = swimEndIndB[nonSuperShort];
    swimStartIndB = swimStartIndB[nonSuperShort];
    
    swimStartIndT = burstBothIndT[swimStartIndB];
    swimStartT = np.zeros(fltCh1.size);
    swimStartT[swimStartIndT] = 1;
    
    swimEndIndT = burstBothIndT[swimEndIndB];
    swimEndT = np.zeros(fltCh1.size);
    swimEndT[swimEndIndT] = 1;

    return swimStartIndT, swimEndIndT