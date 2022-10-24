from scipy.io import loadmat
from h5py import File
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from scipy.stats import zscore
from scipy.stats import spearmanr
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import Rotator
sns.set(style='ticks', font_scale=1.)
plt.close('all')
plt.rcParams["figure.figsize"] = (4, 3)
df = pd.read_csv('../data/datalist.csv', index_col=0)


def ecdf(sample):
    import numpy as np
    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf_ = ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf_(x)
    return x, y