"""Swim probability of paralysed larval zebrafish as a function of the oxygen level and the swim history.

Everything the two demonstration notebooks need is in this file (nothing is imported from the rest of the repository):

    1. data           O2 and swim bins (1 s) of a recording, normoxia / hypoxia windows, the inputs of the models
    2. models         five logistic models with swim and O2 kernels, penalised maximum likelihood, 10-fold cross-validation
    3. scores         R^2 against a pooled intercept-only null, AIC with the effective number of parameters
    4. simulation     free-running simulation of a held-out block and transition matrices P(state(t+k) | state(t))

Conventions
    * time is binned in 1 s bins; bin t covers [t, t + 1) s of the recording
    * label y[t] = 1 if at least one swim STARTS in bin t
    * swim input of a bin = sqrt(summed normalised power of the swims that start in it) (rescaled per fish, see prepare_fish)
    * O2 input = the O2 of the fish z-scored over its normoxia and hypoxia bins (O2 is normalised to the mean of minutes 5-10 = 1)
    * kernels: swim lags 1 ... n_lags, O2 lags 0 ... n_lags - 1 (in bins)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from h5py import File
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.signal import medfilt
from scipy.stats import wilcoxon

SAMPLE_RATE_HZ = 6000
CONDITIONS = ('normoxia', 'hypoxia')

# the five models of the report: swim = kernel of the past swims (None | 'shared' one kernel for both conditions | 'own' one per condition |
# 'prev' only the previous bin), o2 = kernel of the oxygen (None | 'cur_own' current bin, own weight per condition | 'own' history, own kernel per
# condition), icpt = 'shared' (one intercept) | 'own' (one per condition). 'fuse' ties the O2 lag-0 weights of the two conditions.
MODEL_NAMES = ['own Current O2', 'own O2 history', 'own O2 history + shared previous swim', 'shared Swim history', 'own Full model, own intercept, fused']
LABELS = {'own Current O2': 'Current O2', 'own O2 history': 'O2 history', 'own O2 history + shared previous swim': 'O2 history + previous swim',
          'shared Swim history': 'Swim history', 'own Full model, own intercept, fused': 'Full model'}
FULL_MODEL = 'own Full model, own intercept, fused'


@dataclass(frozen=True)
class Settings:
    """All the settings of the analysis (the defaults are the ones of the reported fits)."""
    n_lags: int = 25                       # kernel length: swim lags 1..n_lags, O2 lags 0..n_lags-1
    lambda_swim_grid: tuple = (50.0,)      # strength of the swim-kernel smoothness penalty (a grid is searched by inner cross-validation; one value = fixed)
    lambda_o2_grid: tuple = (1e5,)         # strength of the L2 penalty of the O2 history (lags 1 ...; lag 0 is not penalised)
    lambda_fuse: float = 40.0              # tying of the O2 lag-0 weights of the two conditions in the Full model
    strength_model: str = FULL_MODEL       # the model in which the strengths are chosen (then used for all the models)
    n_folds: int = 10                      # contiguous blocks per condition
    purge_s: int = 5                       # bins on each side of a held-out block that are removed from the training bins
    edge_s: float = 0.02                   # a swim that starts in the last edge_s seconds of a bin is counted in the next bin
    mask_occupancy: float = 0.5            # a bin is no target if swims of earlier bins cover more than this fraction of it and no swim starts in it
    min_swim_s: float = 0.04               # shorter swims are removed
    extend_min: float = 2.0                # length (min) of the extra block of data next to each window
    match_min: float = 0.5                 # the extra block is used if this fraction of its bins has O2 in the range of the window
    hypoxia_until_min: float = 40.0        # no hypoxia data after this time (min): the O2 can already be re-oxygenated
    fix_o2_from_shared: bool = True        # O2 sign rule (see fit_model)


CFG = Settings()


def make_models(cfg: Settings = CFG) -> dict:
    return {'own Current O2': dict(swim=None, o2='cur_own', icpt='shared'),
            'own O2 history': dict(swim=None, o2='own', icpt='shared'),
            'own O2 history + shared previous swim': dict(swim='prev', o2='own', icpt='shared'),
            'shared Swim history': dict(swim='shared', o2=None, icpt='shared'),
            FULL_MODEL: dict(swim='own', o2='own', icpt='own', fuse=cfg.lambda_fuse)}


# the reference of the O2 sign rule (fitted only when the rule is needed, never reported)
SHARED_FULL = dict(swim='shared', o2='shared', icpt='shared')


# ==================================================================================================================================
# 1. DATA
# ==================================================================================================================================
def read_swims(root):
    """Normalised swim power (6 kHz) and the start / end samples of the detected swims of one recording.

    power = fltCh2 - back2 (negative values set to 0), divided by the 99.99th percentile of its positive values."""
    d = File(Path(root) / 'data.mat', 'r')['data']
    g = lambda k: np.asarray(d[k][()]).squeeze()
    power = g('fltCh2') - g('back2')
    power[power < 0] = 0
    power = power / np.percentile(power[power > 0], 99.99)
    return power, g('swimStartIndT').astype(int), g('swimEndIndT').astype(int)


# recordings whose swim record starts this many seconds AFTER the O2 record (key = folder name of the recording); their O2 is cropped in oxygen_bins
SWIM_OFFSET_S = {'fish03-7dpf': 1000}
# recordings without the extra block of matched O2 next to the windows (the window of the offset recording is the whole usable period)
NO_EXTENSION = ('fish03-7dpf',)


def oxygen_bins(root, t_end_min=60.0, bin_s=1):
    """O2 in bins of bin_s seconds, normalised to the mean of minutes 5-10 of the recording (= 1)."""
    ox = np.asarray(File(Path(root) / 'O2_real.mat', 'r')['O2_real'][()]).squeeze().astype(float)
    ox = ox / ox[SAMPLE_RATE_HZ * 60 * 5:SAMPLE_RATE_HZ * 60 * 10].mean()
    ox = ox[int(SWIM_OFFSET_S.get(Path(root).name, 0) * SAMPLE_RATE_HZ):]      # a recording whose swim record starts later than its O2 record: crop (after the normalisation) so that O2 and swims share one time axis
    n = int(bin_s * SAMPLE_RATE_HZ)
    ox = ox[:int(t_end_min * 60 * SAMPLE_RATE_HZ)]
    return ox[:len(ox) // n * n].reshape(-1, n).mean(axis=1)


def swim_bins(root, cfg: Settings = CFG, t_end_min=60.0):
    """Per 1 s bin: count = number of swims that START in the bin, vigor = sqrt(summed power of those swims, from start to end, / samples per bin),
    cover = fraction of the bin covered by swims that started in an earlier bin.

    A swim that starts in the last cfg.edge_s seconds of a bin is counted in the next bin. Swims shorter than cfg.min_swim_s are removed."""
    sr = SAMPLE_RATE_HZ
    power, st, en = read_swims(root)
    nf = min(int(t_end_min * 60 * sr), len(power))
    power = power[:nf]
    nbin = nf // sr
    cs = np.concatenate([[0.0], np.cumsum(power)])
    keep = st < nf
    st, en = st[keep], np.minimum(en[keep], nf)
    keep = (en - st) >= cfg.min_swim_s * sr
    st, en = st[keep], en[keep]
    count, psum, cover = np.zeros(nbin), np.zeros(nbin), np.zeros(nbin)
    for a, b in zip(st, en):
        kb = a // sr + int((a % sr) >= (1 - cfg.edge_s) * sr)      # the bin where the swim is counted
        if kb < nbin:
            count[kb] += 1
            psum[kb] += (cs[b] - cs[a]) / sr
        for k in range(kb + 1, min((b - 1) // sr + 1, nbin)):      # the swim runs on into later bins
            cover[k] += (min(b, (k + 1) * sr) - max(a, k * sr)) / sr
    return count, np.sqrt(psum), cover


def detect_periods(o2, win_min=10, smooth_s=61, t_analysis_min=60, normox_start_min=5, hypox_drop=0.5, floor_frac=0.15, min_len_min=5):
    """Normoxia and hypoxia period of a recording from its O2 trace (1 s bins, relative units), times in minutes.

    hypoxia: O2 falls below hypox_drop x the level of minutes 5-10; the period = the first stretch (>= min_len_min) in which O2 stays in the lowest
    floor_frac of the drop, at most win_min long. normoxia: from normox_start_min while O2 stays within floor_frac of the drop below the level of
    minutes 5-10, at most win_min long. Returns dict(normoxia=(lo, hi) | None, hypoxia=(lo, hi) | None, info=...)."""
    o = np.asarray(o2, float)[:int(t_analysis_min * 60)]
    s = medfilt(o, smooth_s)
    tm = np.arange(len(s)) / 60
    info = {}
    if tm[-1] < 10: return dict(normoxia=None, hypoxia=None, info=dict(reason='recording shorter than 10 min'))
    ref = np.median(s[(tm >= 5) & (tm < 10)])
    floor = s[tm >= 10].min()
    info.update(ref=ref, floor=floor, ratio=floor / ref if ref > 0 else np.nan)
    out = dict(normoxia=None, hypoxia=None, info=info)
    if ref > 0 and floor <= hypox_drop * ref:
        thr = floor + floor_frac * (ref - floor)
        low = (s <= thr) & (tm >= 10)
        segs, i = [], 0
        while i < len(low):
            if low[i]:
                j = i
                while j < len(low) and low[j]: j += 1
                segs.append((tm[i], tm[j - 1] + 1 / 60))
                i = j
            else: i += 1
        segs = [g for g in segs if g[1] - g[0] >= min_len_min]
        if segs:
            a, b = segs[0]
            out['hypoxia'] = (round(a, 2), round(min(b, a + win_min), 2))
            info['hypoxia_segment'] = (round(a, 2), round(b, 2))
    band = floor_frac * (ref - floor) if out['hypoxia'] is not None else 0.25 * ref
    ok = s >= ref - band
    i0 = int(normox_start_min * 60)
    j = i0
    while j < len(ok) and ok[j]: j += 1
    seg = (tm[i0], tm[j - 1] + 1 / 60) if j > i0 else None
    if seg is not None and seg[1] - seg[0] >= min_len_min:
        a = seg[0]
        b = min(seg[1], a + win_min)
        out['normoxia'] = (round(a, 2), round(b, 2))
        info['normoxia_segment'] = (round(seg[0], 2), round(seg[1], 2))
    return out


# periods (min) set by hand: they replace the detected ones (the other period is still detected)
HAND_SET_PERIODS = {'20220730-fish03-TS3': {'normoxia': (3.0, 20.0), 'hypoxia': (30.0, 40.0)},
                    'fish01-6dpf': {'normoxia': (5.0, 20.0)},
                    # fish03-7dpf: minutes of the SWIM record (the O2 record starts 1000 s earlier, SWIM_OFFSET_S); normoxia = the re-normoxia after the hypoxia
                    'fish03-7dpf': {'hypoxia': (13.33, 23.33), 'normoxia': (33.33, 43.33)},
                    # 20220912-fish02 (gtACR2; 0-20 min normal water, 20-40 min low-O2 perfusion): the swim rate is high in 0-8 min; the onset burst of swimming at 26-28 min is left out
                    '20220912-fish02': {'normoxia': (8.0, 20.0), 'hypoxia': (30.0, 40.0)}}


def analysis_windows(root, name='', cfg: Settings = CFG, hand_set=HAND_SET_PERIODS):
    """(lo, hi) in minutes of the normoxia and the hypoxia window of a recording: detected from its O2 trace (detect_periods); the hypoxia window is
    extended up to cfg.hypoxia_until_min while O2 stays low; hand-set periods replace the detected ones; a window must lie inside the swim record."""
    o2 = oxygen_bins(root, 240)
    count, _, _ = swim_bins(root, cfg, t_end_min=240)
    n = min(len(o2), len(count))
    p = detect_periods(o2[:n])
    record_min = len(count) / 60
    if p['hypoxia'] is not None and 'hypoxia_segment' in p['info']:
        seg_end = p['info']['hypoxia_segment'][1]
        p['hypoxia'] = (p['hypoxia'][0], round(min(cfg.hypoxia_until_min, max(p['hypoxia'][1], min(seg_end, cfg.hypoxia_until_min))), 2))
    for c, w in hand_set.get(name, {}).items(): p[c] = tuple(w)
    out = {}
    for c in CONDITIONS:
        w = p[c]
        if w is not None and w[1] > record_min: w = (w[0], round(record_min, 2)) if record_min - w[0] >= 5 else None
        out[c] = w
    return out


def extension_block(o2, lo, hi, condition, cfg: Settings = CFG):
    """An extra block of cfg.extend_min minutes next to the window (lo, hi) whose O2 matches the O2 of the window: the block (outside the window) with
    the largest fraction of bins in the range of the window (5-95 percentile -/+ 1 s.d.), ties to the block closest to the window. Returns
    (start, end) in minutes or None if no block has at least cfg.match_min of its bins in the range."""
    tm = np.arange(len(o2)) / 60
    ow = o2[(tm >= lo) & (tm < hi)]
    sd = ow.std()
    q = (np.percentile(ow, 5) - sd, np.percentile(ow, 95) + sd)
    amin, amax = (5, 25) if condition == 'normoxia' else (20, min(len(o2) / 60, cfg.hypoxia_until_min))
    best = None
    for a in np.arange(amin, amax - cfg.extend_min + 1e-9, 1 / 6):
        b = a + cfg.extend_min
        if b > lo and a < hi: continue
        oe = o2[(tm >= a) & (tm < b)]
        key = (np.mean((oe >= q[0]) & (oe <= q[1])), -min(abs(a - hi), abs(b - lo)))
        if best is None or key > best[0]: best = (key, a, b)
    return (best[1], best[2]) if best is not None and best[0][0] >= cfg.match_min else None


def test_chunks(idx, n_folds, purge_s):
    """Contiguous blocks of the bins idx of one condition: (positions of the held-out block, positions of the training bins) per fold; the training
    bins within purge_s bins of the held-out block are removed."""
    out = []
    for te in np.array_split(np.arange(len(idx)), n_folds):
        lo, hi = idx[te[0]], idx[te[-1]]
        out.append((te, np.flatnonzero((idx < lo - purge_s) | (idx > hi + purge_s))))
    return out


def lag_matrix(x, lags):
    """Column j holds x delayed by lags[j] bins (zero before the start)."""
    n = len(x)
    M = np.zeros((n, len(lags)))
    for j, l in enumerate(lags): M[l:, j] = x[:n - l]
    return M


def prepare_fish(name, root, cfg: Settings = CFG, windows=None):
    """Everything the models need for one recording, as a dict of arrays.

    windows: {'normoxia': (lo, hi), 'hypoxia': (lo, hi)} in minutes, detected from the O2 trace if None.
    The bins of a condition = its window plus an extra block of matched O2 (extension_block), from bin n_lags on, without the bins that only hold the
    continuation of a swim that started earlier.

    Keys: o2 (1 s bins), count (swim starts per bin), sw (swim input), z (z-scored O2), D[condition] = (y, bin indices), iall / yall / call (both
    conditions stacked: bin index, label, condition 0 / 1), folds[condition] (see test_chunks), Ls / Lo (swim / O2 lag matrices), windows, extra."""
    o2 = oxygen_bins(root, 60)
    count, vigor, cover = swim_bins(root, cfg, t_end_min=60)
    n = min(len(o2), len(count))
    o2, count, vigor, cover = o2[:n], count[:n], vigor[:n], cover[:n]
    if windows is None: windows = analysis_windows(root, name, cfg)
    first_bin = cfg.n_lags
    t_min = np.arange(n) / 60
    in_period = lambda lo, hi: (t_min >= lo) & (t_min < hi) & (np.arange(n) >= first_bin)
    drop = (cover > cfg.mask_occupancy) & (count == 0)      # left out as targets (they stay in the swim history)
    D, extra = {}, {}
    for c in CONDITIONS:
        lo, hi = windows[c]
        mask = in_period(lo, hi)
        e = extension_block(o2, lo, hi, c, cfg) if name not in NO_EXTENSION else None
        if e is not None:
            extra[c] = (min(lo, e[0]), max(hi, e[1]))      # window, the gap and the block as one continuous period
            mask = mask | in_period(*extra[c])
        idx = np.flatnonzero(mask & ~drop)
        D[c] = (count[idx] > 0.02, idx)
    iall = np.concatenate([D[c][1] for c in CONDITIONS])
    yall = np.concatenate([D[c][0] for c in CONDITIONS])
    call = np.concatenate([np.full(len(D[c][1]), ci) for ci, c in enumerate(CONDITIONS)])
    z = (o2 - o2[iall].mean()) / o2[iall].std()      # O2 over both windows: centred and scaled, so that one intercept and the O2 input can describe both conditions
    m = count > 0.02
    sw = vigor * (count[m].mean() / vigor[m].mean())      # swim input: the vigor, rescaled per fish to the mean occupancy of its swim bins
    return dict(name=name, n=n, o2=o2, count=count, sw=sw, z=z, D=D, iall=iall, yall=yall, call=call,
                folds={c: test_chunks(D[c][1], cfg.n_folds, cfg.purge_s) for c in CONDITIONS},
                Ls=lag_matrix(sw, range(1, cfg.n_lags + 1)), Lo=lag_matrix(z, range(cfg.n_lags)), windows=windows, extra=extra)


def fold_split(fish, k):
    """Training and held-out bins of fold k: per condition (bin indices, labels), and the stacked training bins (indices, labels, condition id)."""
    D, ch = fish['D'], fish['folds']
    tr = {c: (D[c][1][ch[c][k][1]], D[c][0][ch[c][k][1]]) for c in CONDITIONS}
    te = {c: (D[c][1][ch[c][k][0]], D[c][0][ch[c][k][0]]) for c in CONDITIONS}
    itr = np.concatenate([tr[c][0] for c in CONDITIONS])
    ytr = np.concatenate([tr[c][1] for c in CONDITIONS])
    ctr = np.concatenate([np.full(len(tr[c][0]), ci) for ci, c in enumerate(CONDITIONS)])
    return tr, te, itr, ytr, ctr


# ==================================================================================================================================
# 2. MODELS: design, penalties, penalised logistic regression, cross-validation
# ==================================================================================================================================
def loglik_bins(y, eta):
    """Bernoulli log-likelihood of every bin for the linear predictor eta (logit scale)."""
    return np.where(y, -np.logaddexp(0, -eta), -np.logaddexp(0, eta))


def design(spec, fish, idx, cid):
    """Design matrix of the bins idx (cid = condition id of every bin: 0 normoxia, 1 hypoxia) and its blocks (kind, first column, number of columns).

    Blocks in this order: the intercept(s), the swim kernel(s), the O2 kernel(s); a kernel per condition has its weights multiplied by the
    condition indicator."""
    m = [cid == 0, cid == 1]
    cols, blocks = [], []
    def add(kind, M):
        blocks.append((kind, sum(b[2] for b in blocks), M.shape[1]))
        cols.append(M)
    add('icpt', np.column_stack(m).astype(float) if spec['icpt'] == 'own' else np.ones((len(idx), 1)))
    Ls = fish['Ls'][idx]
    if spec['swim'] == 'shared': add('swim', Ls)
    elif spec['swim'] == 'own': add('swim', Ls * m[0][:, None]); add('swim', Ls * m[1][:, None])
    elif spec['swim'] == 'prev': add('swim', Ls[:, :1])
    if spec['o2']:
        Lo = fish['Lo'][idx]
        if spec['o2'] == 'cur_own': Lo = Lo[:, :1]
        if spec['o2'] in ('own', 'cur_own'): add('o2', Lo * m[0][:, None]); add('o2', Lo * m[1][:, None])
        else: add('o2', Lo)
    return np.hstack(cols), blocks


def penalty_block(n, kind):
    """Penalty matrix of one kernel of n weights: swim kernels 'smooth' (squared second differences + a tiny ridge), O2 kernels plain L2."""
    if kind == 'o2': return np.eye(n)
    D2 = np.diff(np.eye(n), 2, axis=0)
    return D2.T @ D2 + 1e-3 * np.eye(n)


def penalty_matrix(blocks, kind):
    """Penalty of all the kernels of one kind ('swim' or 'o2'); a single-weight kernel gets a tiny ridge; the O2 lag-0 weight is not penalised."""
    P = np.zeros((sum(b[2] for b in blocks),) * 2)
    for k_, s, L in blocks:
        if k_ != kind: continue
        if L == 1:
            P[s, s] = 1e-6
            continue
        a = 1 if kind == 'o2' else 0
        P[s + a:s + L, s + a:s + L] = penalty_block(L - a, kind)
        if a: P[s, s] = 1e-6
    return P


def fuse_matrix(blocks):
    """w' F w = (O2 lag-0 weight of normoxia - O2 lag-0 weight of hypoxia)^2"""
    F = np.zeros((sum(b[2] for b in blocks),) * 2)
    o2 = [b for b in blocks if b[0] == 'o2']
    u, v = o2[0][1], o2[1][1]
    F[u, u] += 1; F[v, v] += 1; F[u, v] -= 1; F[v, u] -= 1
    return F


# convergence of the optimiser: tight, so that the fit (a convex problem) does not depend on the last digits of the inputs
OPTIMISER_OPTIONS = dict(maxiter=20000, maxfun=50000, ftol=1e-14, gtol=1e-8)


def fit_penalised(Z, y, P, w0, bounds=None):
    """Logistic regression: minimise -loglik + 0.5 w' P w (L-BFGS-B, converged to OPTIMISER_OPTIONS)."""
    def f(w):
        eta = Z @ w
        p = 1 / (1 + np.exp(-eta))
        pw = P @ w
        return -loglik_bins(y > 0, eta).sum() + 0.5 * (w @ pw), Z.T @ (p - y) + pw
    return minimize(f, w0, jac=True, method='L-BFGS-B', bounds=bounds, options=OPTIMISER_OPTIONS).x


def start_weights(Z, y, n_icpt):
    p0 = np.clip(y.mean(), 1e-3, 1 - 1e-3)
    w0 = np.zeros(Z.shape[1])
    w0[:n_icpt] = np.log(p0 / (1 - p0))
    return w0


def inner_folds(itr, ctr, n_folds, purge_s, n=5):
    per = [np.flatnonzero(ctr == ci) for ci in (0, 1)]
    ch = [test_chunks(itr[p], n, purge_s) for p in per]
    return [(np.concatenate([per[ci][ch[ci][k][1]] for ci in (0, 1)]), np.concatenate([per[ci][ch[ci][k][0]] for ci in (0, 1)])) for k in range(n)]


def select_strengths(fish, itr, ytr, ctr, cfg: Settings = CFG, models=None):
    """(lambda_swim, lambda_O2) chosen jointly on the grids of cfg by an inner 5-fold contiguous cross-validation of cfg.strength_model on the training
    bins (held-out log-likelihood). One value per grid = no search."""
    if len(cfg.lambda_swim_grid) == 1 and len(cfg.lambda_o2_grid) == 1:
        return {'swim': cfg.lambda_swim_grid[0], 'o2': cfg.lambda_o2_grid[0]}
    models = models or make_models(cfg)
    spec = models[cfg.strength_model]
    Z, blocks = design(spec, fish, itr, ctr)
    Ps, Po = penalty_matrix(blocks, 'swim'), penalty_matrix(blocks, 'o2')
    Pf = spec['fuse'] * fuse_matrix(blocks) if spec.get('fuse') else 0
    yb = ytr.astype(float)
    w0 = start_weights(Z, yb, blocks[0][2])
    folds = inner_folds(itr, ctr, cfg.n_folds, cfg.purge_s)
    best = None
    for ls in cfg.lambda_swim_grid:
        for lo in cfg.lambda_o2_grid:
            P = ls * Ps + lo * Po + Pf
            score = 0.0
            for tr_, te_ in folds:
                w = fit_penalised(Z[tr_], yb[tr_], P, w0)
                score -= loglik_bins(yb[te_] > 0, Z[te_] @ w).sum()
            if best is None or score < best[0]: best = (score, ls, lo)
    return {'swim': best[1], 'o2': best[2]}


def fit_model(fish, spec, itr, ytr, ctr, lam, cfg: Settings = CFG, apply_rule=True):
    """Fit one model on the bins (itr, ytr, ctr). Returns (weights, design, blocks, penalty matrix, fixed O2 lag-0 weight or None).

    O2 sign rule (cfg.fix_o2_from_shared; models with an own intercept, own swim kernels and own O2 kernels): if the mean O2 lag-0 weight is negative
    (more O2 would lower the swim probability), the shared Full model is fitted on the same bins; if ITS O2 lag-0 weight is positive, the O2 lag-0
    weights of the own model are fixed to that value and all the other weights are refitted. The fixed weights do not count as parameters."""
    Z, blocks = design(spec, fish, itr, ctr)
    P = lam['swim'] * penalty_matrix(blocks, 'swim') + lam['o2'] * penalty_matrix(blocks, 'o2')
    if spec.get('fuse'): P = P + spec['fuse'] * fuse_matrix(blocks)
    yb = ytr.astype(float)
    w0 = start_weights(Z, yb, blocks[0][2])
    w = fit_penalised(Z, yb, P, w0)
    fixed = None
    ob = [b for b in blocks if b[0] == 'o2']
    if apply_rule and cfg.fix_o2_from_shared and spec['icpt'] == 'own' and spec['swim'] == 'own' and spec['o2'] in ('own', 'cur_own') and len(ob) == 2 \
            and np.mean([w[b[1]] for b in ob]) < 0:
        w_sh, _, bl_sh, _, _ = fit_model(fish, SHARED_FULL, itr, ytr, ctr, lam, cfg, apply_rule=False)
        v = float(w_sh[[b for b in bl_sh if b[0] == 'o2'][0][1]])
        if v > 0:
            bounds = [(None, None)] * len(w)
            for b in ob:
                bounds[b[1]] = (v, v)
                w0[b[1]] = v
            w = fit_penalised(Z, yb, P, w0, bounds)
            fixed = v
            P = P.copy()
            for b in ob: P[b[1], b[1]] += 1e12
    return w, Z, blocks, P, fixed


def fold_task(fish, k, cfg: Settings = CFG):
    """Fold k: strengths on the training bins, every model fitted on them, log-likelihood of the held-out block per condition.
    Returns (ll[model] = [normoxia, hypoxia], strengths, weights[model] = (w, blocks))."""
    models = make_models(cfg)
    tr, te, itr, ytr, ctr = fold_split(fish, k)
    lam = select_strengths(fish, itr, ytr, ctr, cfg, models)
    ll, W = {}, {}
    for nm, spec in models.items():
        w, _, blocks, _, _ = fit_model(fish, spec, itr, ytr, ctr, lam, cfg)
        W[nm] = (w, blocks)
        ll[nm] = [loglik_bins(te[c][1], design(spec, fish, te[c][0], np.full(len(te[c][0]), ci))[0] @ w).sum() for ci, c in enumerate(CONDITIONS)]
    return ll, lam, W


def full_task(fish, cfg: Settings = CFG):
    """Every model fitted on all the bins of the fish: weights, in-sample log-likelihood (all / normoxia / hypoxia) and the effective number of
    parameters edf = trace[(Z'WZ + P)^-1 Z'WZ] (W = p (1 - p))."""
    models = make_models(cfg)
    lam = select_strengths(fish, fish['iall'], fish['yall'], fish['call'], cfg, models)
    out = {'lam': lam}
    for nm, spec in models.items():
        w, Z, blocks, P, fixed = fit_model(fish, spec, fish['iall'], fish['yall'], fish['call'], lam, cfg)
        eta = Z @ w
        ll = loglik_bins(fish['yall'], eta)
        pr = 1 / (1 + np.exp(-eta))
        H = (Z * (pr * (1 - pr))[:, None]).T @ Z
        out[nm] = dict(o2_fixed=fixed, w=w, blocks=blocks, ll_in=ll.sum(), ll_in_n=ll[fish['call'] == 0].sum(), ll_in_h=ll[fish['call'] == 1].sum(),
                       edf=float(np.trace(np.linalg.solve(H + P + 1e-9 * np.eye(len(w)), H))))
    return out


def run_fits(fishes, cfg: Settings = CFG, n_jobs=64):
    """Cross-validation fits (every fish x fold) and full-data fits (every fish), in parallel. Returns (LF, LFULL): LF[(i, k)] = fold_task output,
    LFULL[i] = full_task output (i = position of the fish in the list)."""
    jobs = [(i, k) for i in range(len(fishes)) for k in range(cfg.n_folds)]
    lf = Parallel(n_jobs=n_jobs, backend='loky')(delayed(fold_task)(fishes[i], k, cfg) for i, k in jobs)
    lfull = Parallel(n_jobs=n_jobs, backend='loky')(delayed(full_task)(f, cfg) for f in fishes)
    return dict(zip(jobs, lf)), dict(enumerate(lfull))


# ==================================================================================================================================
# 3. SCORES: R^2 against the pooled null, AIC
# ==================================================================================================================================
def null_loglik(fish, cfg: Settings = CFG):
    """Log-likelihood of the pooled intercept-only null (ONE swim rate for both conditions).
    Held-out: the rate of the stacked training bins of a fold, scored on the held-out block of each condition (summed over the folds).
    In-sample: the rate of all the bins of the fish. Returns (held-out [normoxia, hypoxia], in-sample [normoxia, hypoxia])."""
    held = [0.0, 0.0]
    for k in range(cfg.n_folds):
        tr, te, _, ytr, _ = fold_split(fish, k)
        p0 = np.clip(np.mean(ytr), 1e-6, 1 - 1e-6)
        for ci, c in enumerate(CONDITIONS): held[ci] += loglik_bins(te[c][1], np.full(len(te[c][1]), np.log(p0 / (1 - p0)))).sum()
    pa = np.clip(fish['yall'].mean(), 1e-6, 1 - 1e-6)
    insample = [loglik_bins(fish['D'][c][0], np.full(len(fish['D'][c][0]), np.log(pa / (1 - pa)))).sum() for c in CONDITIONS]
    return held, insample


def score_table(fishes, LF, LFULL, cfg: Settings = CFG):
    """One row per fish and model: R^2 (in-sample and held-out) = 1 - loglik(model) / loglik(pooled null), AIC = 2 edf - 2 loglik, edf, the O2 lag-0 weights
    of the models with an own O2 kernel. Columns for both conditions: pooled over all the bins (r2_in, r2_cv, aic: a condition with more bins counts more),
    per condition (suffix _n, _h) and balanced (suffix _bal: both conditions count equally: the mean of the two R^2, the AIC per bin)."""
    rows = []
    for i, fish in enumerate(fishes):
        nul, nuli = null_loglik(fish, cfg)
        for nm in MODEL_NAMES:
            f = LFULL[i][nm]
            ll = np.sum([LF[(i, k)][0][nm] for k in range(cfg.n_folds)], axis=0)
            ob = [b for b in f['blocks'] if b[0] == 'o2']
            wn, wh = (f['w'][ob[0][1]], f['w'][ob[1][1]]) if len(ob) == 2 else (np.nan, np.nan)
            r2_in_n, r2_in_h = 1 - f['ll_in_n'] / nuli[0], 1 - f['ll_in_h'] / nuli[1]
            r2_cv_n, r2_cv_h = 1 - ll[0] / nul[0], 1 - ll[1] / nul[1]
            n_n, n_h = len(fish['D']['normoxia'][0]), len(fish['D']['hypoxia'][0])      # bins of the two conditions
            aic_n, aic_h = 2 * f['edf'] - 2 * f['ll_in_n'], 2 * f['edf'] - 2 * f['ll_in_h']
            rows.append(dict(fish=fish['name'], model=nm, edf=f['edf'], aic=2 * f['edf'] - 2 * f['ll_in'], aic_n=aic_n, aic_h=aic_h,
                             # balanced AIC = AIC per bin: mean over the two conditions of (-2 loglik of the condition / its bins) + 2 edf / all bins
                             aic_bal=((aic_n - 2 * f['edf']) / n_n + (aic_h - 2 * f['edf']) / n_h) / 2 + 2 * f['edf'] / (n_n + n_h),
                             r2_in=1 - f['ll_in'] / sum(nuli), r2_in_n=r2_in_n, r2_in_h=r2_in_h, r2_in_bal=(r2_in_n + r2_in_h) / 2,
                             r2_cv=1 - ll.sum() / sum(nul), r2_cv_n=r2_cv_n, r2_cv_h=r2_cv_h, r2_cv_bal=(r2_cv_n + r2_cv_h) / 2, w0_n=wn, w0_h=wh))
    return pd.DataFrame(rows)


def kernels_of(w, blocks, condition):
    """Intercept, swim kernel (lags 1 ...) and O2 kernel (lags 0 ...) of a fitted model for one condition (0 normoxia, 1 hypoxia); None if absent."""
    get = lambda kind: [w[s:s + L] for (k_, s, L) in blocks if k_ == kind]
    b0, sw, o2 = get('icpt')[0], get('swim'), get('o2')
    pick = lambda lst: (lst[condition] if len(lst) == 2 else lst[0]) if lst else None
    return (b0[condition] if len(b0) == 2 else b0[0]), pick(sw), pick(o2)


def save_results(path, cfg, fishes, LF, LFULL):
    with open(path, 'wb') as fh:
        pickle.dump(dict(settings=asdict(cfg), fishes=fishes, LF=LF, LFULL=LFULL), fh)


def load_results(path):
    with open(path, 'rb') as fh:
        r = pickle.load(fh)
    return Settings(**r['settings']), r['fishes'], r['LF'], r['LFULL']


def repeated_measures_anova(X):
    """One-way repeated-measures (within-subject) ANOVA. X: array (subjects x conditions), here fish x models.

    F = MS(conditions) / MS(conditions x subjects), df = (k - 1, (n - 1)(k - 1)). Also returns the Greenhouse-Geisser epsilon (sphericity correction,
    1 = sphericity holds) and the p-value with the degrees of freedom multiplied by it. Returns dict(F, df1, df2, p, epsilon, p_gg)."""
    from scipy.stats import f as f_dist
    X = np.asarray(X, float)
    n, k = X.shape
    gm = X.mean()
    ss_cond = n * ((X.mean(0) - gm) ** 2).sum()
    ss_subj = k * ((X.mean(1) - gm) ** 2).sum()
    ss_err = ((X - gm) ** 2).sum() - ss_cond - ss_subj
    df1, df2 = k - 1, (n - 1) * (k - 1)
    F = (ss_cond / df1) / (ss_err / df2)
    S = np.cov(X, rowvar=False)
    H = np.eye(k) - 1.0 / k
    Sc = H @ S @ H
    eps = float(np.trace(Sc) ** 2 / ((k - 1) * (Sc ** 2).sum()))
    return dict(F=F, df1=df1, df2=df2, p=float(f_dist.sf(F, df1, df2)), epsilon=eps, p_gg=float(f_dist.sf(F, eps * df1, eps * df2)))


def _greenhouse_geisser(X):
    """Greenhouse-Geisser epsilon of the columns of X (subjects x levels): 1 = sphericity holds, 1 / (levels - 1) = the lowest value."""
    k = X.shape[1]
    H = np.eye(k) - 1.0 / k
    Sc = H @ np.atleast_2d(np.cov(X, rowvar=False)) @ H
    return float(np.trace(Sc) ** 2 / ((k - 1) * (Sc ** 2).sum()))


def repeated_measures_anova_2way(X):
    """Two-way within-subject (repeated-measures) ANOVA with the factors A (e.g. model) and B (e.g. oxygen condition). X: array (subjects, A, B).

    Every effect is tested against its own interaction with the subjects (A against A x subjects, B against B x subjects, A x B against A x B x subjects).
    Returns {'A': ..., 'B': ..., 'AxB': ...}, each dict(F, df1, df2, p, epsilon, p_gg); epsilon / p_gg = Greenhouse-Geisser correction (for B, with two
    levels, epsilon = 1; for A x B it is computed from the A-profile of the difference between the two B levels when B has two levels)."""
    from scipy.stats import f as f_dist
    X = np.asarray(X, float)
    n, a, b = X.shape
    gm = X.mean()
    m_s, m_a, m_b = X.mean((1, 2)), X.mean((0, 2)), X.mean((0, 1))
    m_sa, m_sb, m_ab = X.mean(2), X.mean(1), X.mean(0)
    ss_a = n * b * ((m_a - gm) ** 2).sum()
    ss_b = n * a * ((m_b - gm) ** 2).sum()
    ss_ab = n * ((m_ab - m_a[:, None] - m_b[None, :] + gm) ** 2).sum()
    ss_as = b * ((m_sa - m_s[:, None] - m_a[None, :] + gm) ** 2).sum()
    ss_bs = a * ((m_sb - m_s[:, None] - m_b[None, :] + gm) ** 2).sum()
    resid = X - m_sa[:, :, None] - m_sb[:, None, :] - m_ab[None] + m_s[:, None, None] + m_a[None, :, None] + m_b[None, None, :] - gm
    ss_abs = (resid ** 2).sum()
    def effect(ss, df1, ss_err, df2, eps):
        F = (ss / df1) / (ss_err / df2)
        return dict(F=F, df1=df1, df2=df2, p=float(f_dist.sf(F, df1, df2)), epsilon=eps, p_gg=float(f_dist.sf(F, eps * df1, eps * df2)))
    eps_a = _greenhouse_geisser(m_sa)
    eps_ab = _greenhouse_geisser(X[:, :, 0] - X[:, :, 1]) if b == 2 else _greenhouse_geisser(X.reshape(n, -1))
    return {'A': effect(ss_a, a - 1, ss_as, (a - 1) * (n - 1), eps_a),
            'B': effect(ss_b, b - 1, ss_bs, (b - 1) * (n - 1), 1.0 if b == 2 else _greenhouse_geisser(m_sb)),
            'AxB': effect(ss_ab, (a - 1) * (b - 1), ss_abs, (a - 1) * (b - 1) * (n - 1), eps_ab)}


def compare_with_reference(X, names, reference):
    """Paired comparison of every column of X (subjects x models) with the reference column: mean, mean difference (model - reference), paired t-test
    and Wilcoxon signed-rank p-value over the subjects (uncorrected). Returns a DataFrame with one row per model except the reference."""
    from scipy.stats import ttest_rel
    X = np.asarray(X, float)
    r = list(names).index(reference)
    rows = []
    for j, nm in enumerate(names):
        if j == r: continue
        d = X[:, j] - X[:, r]
        rows.append({'model': nm, 'mean': X[:, j].mean(), 'mean - reference': d.mean(), 'p (paired t-test)': ttest_rel(X[:, j], X[:, r]).pvalue, 'p (Wilcoxon)': paired_wilcoxon(d)})
    return pd.DataFrame(rows).set_index('model')


def paired_wilcoxon(d):
    """p-value of the Wilcoxon signed-rank test of the paired differences d (nan if all zero)."""
    d = np.asarray(d, float)
    return wilcoxon(d).pvalue if (d != 0).any() else np.nan


# ==================================================================================================================================
# 4. FREE-RUNNING SIMULATION and transition matrices
# ==================================================================================================================================
sigmoid = lambda x: 1 / (1 + np.exp(-x))


def lagged_sum(x, kernel, start):
    """out[t] = sum_j kernel[j] * x[t - start - j]"""
    out = np.zeros(len(x))
    for j, kj in enumerate(kernel):
        l = start + j
        out[l:] += kj * x[:len(x) - l]
    return out


def model_parts(w, blocks, z):
    """What a free run needs: per condition the O2 term of every bin, the swim kernel and the intercept."""
    b0, sw, o2 = [], [], []
    for c in (0, 1):
        i0, s, o = kernels_of(w, blocks, c)
        b0.append(i0); sw.append(s if s is not None else np.zeros(0)); o2.append(lagged_sum(z, o, 0) if o is not None else np.zeros(len(z)))
    return o2, sw, b0


def free_run(parts, sw_obs, pool, cond_of_bin, lo, hi, n_sim, rng):
    """The model generates the bins lo ... hi by itself, n_sim sequences at once. The real swim input is used only for the bins BEFORE lo. In every
    bin a swim is drawn with p = sigmoid(eta) (eta from the weights, the model's OWN past swims and the measured O2); a simulated swim gets a swim
    input drawn from pool[condition] (the swim inputs of the training bins). Returns the drawn swims and the swim probabilities (n_sim x bins)."""
    o2t, kern, b0 = parts
    S = np.zeros((n_sim, hi + 1))
    S[:, :lo] = sw_obs[:lo]
    Y, P = np.zeros((n_sim, hi - lo + 1)), np.zeros((n_sim, hi - lo + 1))
    for u in range(lo, hi + 1):
        c = cond_of_bin[u]
        La = len(kern[c])
        eta = b0[c] + o2t[c][u] + (S[:, u - La:u][:, ::-1] @ kern[c] if La else 0.0)
        p = sigmoid(eta)
        y = rng.random(n_sim) < p
        Y[:, u - lo], P[:, u - lo] = y, p
        S[:, u] = y * rng.choice(pool[c], n_sim)
    return Y, P


def condition_of_bins(fish):
    """The condition (0 / 1) of every bin of the recording (bins outside the windows get the condition of the nearest bin used)."""
    ii = np.concatenate([fish['D'][c][1] for c in CONDITIONS])
    cc = np.concatenate([np.full(len(fish['D'][c][1]), ci) for ci, c in enumerate(CONDITIONS)])
    o = np.argsort(ii)
    ii, cc = ii[o], cc[o]
    t = np.arange(fish['n'])
    near = np.clip(np.searchsorted(ii, t), 1, len(ii) - 1)
    near = np.where(np.abs(ii[near] - t) < np.abs(ii[near - 1] - t), near, near - 1)
    return cc[near]


def training_pool(fish, itr, ytr, ctr):
    """Swim inputs of the swim bins of the training data, per condition (all conditions if a condition has fewer than 10)."""
    pool = []
    for ci in (0, 1):
        p = fish['sw'][itr[(ytr > 0) & (ctr == ci)]]
        p = p[p > 0]
        pool.append(p if len(p) >= 10 else fish['sw'][itr[ytr > 0]])
    return pool


def pair_stats(Y, horizon):
    """For every lag k = 1 ... horizon and state s of bin t (0 pause, 1 swim): [number of pairs (t, t + k), number of swims at t + k]."""
    acc = np.zeros((2, horizon, 2))
    Y = np.atleast_2d(Y)
    for k in range(1, horizon + 1):
        if Y.shape[1] <= k: continue
        x, y = Y[:, :-k], Y[:, k:]
        for s in (0, 1):
            m = x == s
            acc[s, k - 1] = [m.sum(), y[m].sum()]
    return acc


def simulate_fish(fish, fold_weights, cfg: Settings = CFG, horizon=10, n_sim=100, seed=0):
    """Free runs of every held-out block (every fold and condition) with every model, and the real block.

    fold_weights[k][model] = (w, blocks) (the fits of fold k). Returns dict(real, model): pair statistics (see pair_stats) summed over the blocks of a
    condition: real[condition, state, lag - 1, :], model[name][condition, state, lag - 1, :]; the simulated sequences count n_sim times."""
    rng = np.random.default_rng(seed)
    n = fish['n']
    cond_of_bin = condition_of_bins(fish)
    real = np.zeros((2, 2, horizon, 2))
    sim = {nm: np.zeros((2, 2, horizon, 2)) for nm in MODEL_NAMES}
    for k in range(cfg.n_folds):
        tr, te, itr, ytr, ctr = fold_split(fish, k)
        pool = training_pool(fish, itr, ytr, ctr)
        for ci, c in enumerate(CONDITIONS):
            target = te[c][0]
            lo, hi = int(target.min()), int(target.max())
            real[ci] += pair_stats((fish['count'][lo:hi + 1] > 0).astype(float), horizon)
            for nm in MODEL_NAMES:
                w, blocks = fold_weights[k][nm]
                Y, _ = free_run(model_parts(w, blocks, fish['z']), fish['sw'], pool, cond_of_bin, lo, hi, n_sim, rng)
                sim[nm][ci] += pair_stats(Y, horizon)
    return dict(real=real, model=sim)


def run_simulations(fishes, LF, cfg: Settings = CFG, horizon=10, n_sim=100, n_jobs=64):
    """simulate_fish for every fish in parallel (the seed is the position of the fish)."""
    return Parallel(n_jobs=min(len(fishes), n_jobs), backend='loky')(
        delayed(simulate_fish)(f, {k: LF[(i, k)][2] for k in range(cfg.n_folds)}, cfg, horizon, n_sim, i) for i, f in enumerate(fishes))


def transition_matrix(stats, ci, k):
    """Column-stochastic matrix T[a, b] = P(state(t + k) = a | state(t) = b), states ordered (swim, pause), from pair statistics
    stats[condition, state, lag - 1, :]; None if there are fewer than 20 pairs for a state (stats of the real data)."""
    r = stats[ci, :, k - 1]
    p_pause, p_swim = r[0, 1] / max(r[0, 0], 1), r[1, 1] / max(r[1, 0], 1)      # P(swim at t + k | pause at t), P(swim at t + k | swim at t)
    return np.array([[p_swim, p_pause], [1 - p_swim, 1 - p_pause]])


def transition_errors(SIM, ci, k, min_pairs=20):
    """Per fish and model: the mean absolute difference between the model's and the real transition matrix (mean of its four entries), for the fish
    with at least min_pairs real pairs for both states. Returns {model: array over the fish} and the list of fish positions."""
    use = [i for i, s in enumerate(SIM) if s['real'][ci, 0, k - 1, 0] >= min_pairs and s['real'][ci, 1, k - 1, 0] >= min_pairs]
    out = {nm: np.array([np.abs(transition_matrix(SIM[i]['model'][nm], ci, k) - transition_matrix(SIM[i]['real'], ci, k)).mean() for i in use]) for nm in MODEL_NAMES}
    return out, use
