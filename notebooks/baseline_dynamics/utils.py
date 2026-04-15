import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, zscore
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.baseline_dynamics import ecdf, load_baseline_datalist, load_oxygen_mean


sns.set(style="ticks", font_scale=1.0)
plt.close("all")
plt.rcParams["figure.figsize"] = (4, 3)
df = load_baseline_datalist()

__all__ = [
    "df",
    "ecdf",
    "load_baseline_datalist",
    "load_oxygen_mean",
    "mannwhitneyu",
    "np",
    "pd",
    "plt",
    "sns",
    "spearmanr",
    "tqdm",
    "zscore",
]
