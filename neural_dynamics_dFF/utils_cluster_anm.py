import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
from scipy.stats import spearmanr, zscore
sns.set(style='ticks', font_scale=1.)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter
from itertools import count

from sklearn.decomposition import FactorAnalysis

def butter_lowpass_filter(data, cutoff, fs=1, order=5):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def f_FA(Y, L):
    from scipy.linalg import lstsq
    return lstsq(L, Y)[0]


def L_FA(Y, f):
    from scipy.linalg import lstsq
    return lstsq(f.T, Y.T)[0].T


def smooth(a, kernel):
    return np.convolve(a, kernel, 'full')[kernel.shape[0]//2:-(kernel.shape[0]//2)]
    
def gaussKernel(sigma=20):
    kernel = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(np.arange(-sigma*3,sigma*3+1)**2)/(2*sigma**2))
    return kernel/kernel.sum()

