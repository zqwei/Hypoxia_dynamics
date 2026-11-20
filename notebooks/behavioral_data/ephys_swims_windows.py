"""Swims (swimPower1, onset) and normoxia/hypoxia windows of the 29 fictive-ephys recordings; built from swim_vigor_fictive_ephys_v1.ipynb cells 1,3,5,7. Import with contextlib.redirect_stdout to silence."""
from IPython.display import display as _d
display = lambda *a, **k: None

# ---- ephys notebook cell 1
import os
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[v] = '1'
from pathlib import Path
from glob import glob
import warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import ttest_rel, wilcoxon
from h5py import File
warnings.filterwarnings('ignore')

DATA_DIR = Path('/nearline/ahrens/rongwei/nrs_data/data_fictive_ephys')
SR = 6000                                                     # sampling rate (Hz) of the recordings
WINDOWS = {'normoxia': (5.0, 20.0), 'hypoxia': (25.0, 40.0)}  # minutes, the same for every recording
COND = list(WINDOWS); CC = {'normoxia': '#666666', 'hypoxia': '#99ccff'}
FILES = sorted(glob(str(DATA_DIR / '*' / 'ephys' / 'analysis' / 'data.mat')))


# ---- ephys notebook cell 3
def kind(name):
    n = name.lower()
    return ', '.join([k for k, w in (('OMR', 'omr'), ('closed loop', '-cl'), ('open loop', '-ol'), ('spontaneous', 'spon'), ('propofol', 'propofol'), ('nodose ablated', 'nodose'), ('O2 measure', 'o2-measure')) if w in n]) or '?'
SWIMS = []; REC = []
for fn in FILES:
    name = Path(fn).parents[2].name
    with File(fn, 'r') as f:
        d = f['data']; g = lambda k: np.asarray(d[k][()]).squeeze().astype(float)
        st, en, pw = g('swimStartIndT'), g('swimEndIndT'), g('swimPower1'); n_samples = d['ch1'].shape[-1]      # the swim on and off (sample index) and the swim power (one value per swim)
    assert len(st) == len(en) == len(pw), (name, len(st), len(en), len(pw))
    REC.append(dict(recording=name, minutes=n_samples / SR / 60, swims=len(st), kind=kind(name)))
    SWIMS.append(pd.DataFrame(dict(recording=name, onset_min=st / SR / 60, duration_s=(en - st) / SR, vigor=pw)))
SW = pd.concat(SWIMS, ignore_index=True); REC = pd.DataFrame(REC).set_index('recording')
display(REC.round(1)); print('all swims:', len(SW), '| NaN vigor:', int(SW.vigor.isna().sum()))

# ---- ephys notebook cell 5
GROUPS = {      # the recordings of the three groups (folder names under DATA_DIR; a name with a date in front is the one with that date at the end)
    'spontaneous': [
        'fish03-10dpf-huc-ATPsensor-ZTS1-oxygen-spon_20210919_214825',
        'fish02-6dpf-gfap-rgeco-nodose-ablated-ZTS1-oxygen-spon_20220214_002527',
        'fish01-13dpf-gfap-rgeco-ZTS1-oxygen-spon_20220211_155824',
        '20221230/fish01-6dpf-gfap-gc6f-huc-jrgeco1b-TS1-oxygen-spon',
        'fish01-9dpf-gfap-gc6f-huc-jrgeco-ZTS1-oxygen-spon_20220225_121140',
        'fish01-6dpf-huc-grabNE-gfap-jRGECO1b-ZTS1-oxygen-CL_20221216_134714',
        'fish01-6dpf-gfap-gc6f-bARK-D110-TS2-spon-oxygen'],
    'OL': [
        'fish03-chat-gc8f-gfap-jrgeco1b-TS1-oxygen-OMR_20220610_224643',
        'fish02-5dpf-gad1b-gc6f-gfap-jrgeco1b-ZTS1-oxygen-OMR_20220614_174108',
        'fish03-5dpf-gad1b-gc6f-gfap-jrgeco1b-ZTS1-oxygen-OMR_20220614_225237',
        'fish03-5dpf-gad1b-gc6f-gfap-jrgeco1b-TS2-oxygen-OMR_20220615_000604',
        'fish01-6dpf-huc-h2b-gc7f-ZTS1-oxygen-OMR_20220617_184428',
        'fish02-6dpf-huc-h2b-gc7f-ZTS1-oxygen-OMR_20220617_202209',
        'fish04-6dpf-gfap-jrgeco1b-ZTS1-oxygen-OMR_20220625_003742',
        'fish01-gfap-ATPsensor-gfap-rgeco-ZTS1-oxygen-OMR_20220810_222545',
        'fish01-gfap-ATPsensor-gfap-rgeco-ZTS2-oxygen-OMR_20220810_233011',
        'fish01-6dpf-gfap-gc6f-TH1-NTR-ZTS1-oxygen-OMR_20220826_122800',
        'fish02-6dpf-huc-h2b-gc7f-vglut2a-DsRed-ZTS2-oxygen-OL_20220906_215447',
        'fish02-6dpf-huc-h2b-gc7f-ZTS1-oxygen-OMR-propofol_20220916_231529',
        'fish01-7dpf-huc-h2b-gc7f-gfap-jRGECO1b-ZTS1-OL-oxygen_20221118_151457',
        'fish03-7dpf-huc-h2b-gc7f-gfap-jRGECO1b-ZTS1-O2-measure_20221118_232325',
        '20221230/fish03-6dpf-gfap-gc6f-huc-jrgeco1b-TS1-oxygen-OL',
        'fish01-7dpf-huc-iATPsnFR-ZTS2-oxygen-OL_20230127_145653'],
    'CL': [
        'fish01-7dpf-th1-gc6f-ZTS1-oxygen-CL-OL_20220714_124356',
        'fish01-7dpf-TH1-gc6f-ZTS1-oxygen-CL-OL_20220721_110025',
        'fish01-6dpf-huc-h2b-gc7f-vglut2a-DsRed-ZTS1-oxygen-CL_20220906_163404',
        'fish02-6dpf-huc-h2b-gc7f-vglut2a-DsRed-ZTS1-oxygen-CL_20220906_205239',
        'fish01-7dpf-huc-grabNE-ZTS1-CL-oxygen_20221027_162807',
        'fish02-6dpf-gfap-gc6f-ZTS1-CL-oxygen_20221116_213418']}
def _match(name, folders):
    if '/' in name: date, rest = name.split('/'); return [f for f in folders if f.startswith(rest) and f.endswith(date)]
    return [f for f in folders if f == name or f.startswith(name)]
all_folders = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
GROUP_OF = {}
for g, names in GROUPS.items():
    for n in names:
        m = _match(n, all_folders); assert len(m) == 1, (g, n, m); GROUP_OF[m[0]] = g
REC['group'] = [GROUP_OF.get(r) for r in REC.index]
print('recordings with a data.mat that are in none of the groups (not used):'); print(list(REC.index[REC.group.isna()]))
print('listed in a group but without a data.mat (nothing to load):'); print([f for f in GROUP_OF if f not in REC.index])
USED = list(REC.index[REC.group.notna()]); REC = REC.loc[USED]; SW = SW[SW.recording.isin(USED)].reset_index(drop=True)
print({g: int((REC.group == g).sum()) for g in GROUPS}, 'recordings used per group;', len(USED), 'in total')
GCOL = {'spontaneous': '#8c564b', 'OL': '#1f77b4', 'CL': '#d62728'}
def bar_groups(df, cols, ylabel):      # df: one row per recording with the columns of the two windows (cols) and 'group'; one panel per group
    fig, axs = plt.subplots(1, len(GROUPS), figsize=(3.4 * len(GROUPS), 4), sharey=True, squeeze=False)
    for ax, g in zip(axs[0], GROUPS):
        d = df[df.group == g][list(cols)].dropna(); d.columns = COND
        for j, c in enumerate(COND): ax.bar(j, d[c].mean(), width=0.6, color=CC[c] if c == 'hypoxia' else '0.55', alpha=0.6, yerr=d[c].std() / np.sqrt(len(d)), capsize=4, ecolor='k', zorder=1)
        for _, r in d.iterrows(): ax.plot([0, 1], [r[COND[0]], r[COND[1]]], 'o-', color='k', lw=0.6, ms=3, zorder=3)
        ax.set_xticks(range(len(COND))); ax.set_xticklabels(COND); ax.set_title(f'{g} (n = {len(d)})', fontsize=9)
        print(f'{g:12s} n = {len(d):2d} | mean {d[COND[0]].mean():.3f} vs {d[COND[1]].mean():.3f} | median {d[COND[0]].median():.3f} vs {d[COND[1]].median():.3f} | higher in {COND[1]}: {int((d[COND[1]] > d[COND[0]]).sum())} of {len(d)} | paired Wilcoxon p = {wilcoxon(d[COND[0]], d[COND[1]]).pvalue:.4f}')
    axs[0, 0].set_ylabel(ylabel, fontsize=8); sns.despine(fig=fig); plt.tight_layout(); plt.show()

# ---- ephys notebook cell 7
XLSX = next(p for p in (Path('Fig1_ephys_20250722.xlsx'), Path('/groups/ahrens/home/weiz/Projects/Hypoxia_dynamics/notebooks/behavioral_data/Fig1_ephys_20250722.xlsx')) if p.exists())
SHEET = 'choosed data'      # one recording = a row with its date and name, then rows with the state (norm / high / low) and the sweeps of the state; one sweep = 30 s
import zipfile, re, xml.etree.ElementTree as ET
def read_sheet(path, sheet):      # the cells of one sheet of an .xlsx (standard library only): {row: {column number: value}}
    M = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'; R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
    z = zipfile.ZipFile(path); ss = [''.join(t.text or '' for t in si.iter(M + 't')) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).iter(M + 'si')]
    rid = {r.get('Id'): r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    target = [rid[s.get(R)] for s in ET.fromstring(z.read('xl/workbook.xml')).find(M + 'sheets') if s.get('name') == sheet][0]
    def col(c): n = 0; [n := n * 26 + ord(ch) - 64 for ch in c]; return n
    cells = {}
    for c in ET.fromstring(z.read('xl/' + target)).iter(M + 'c'):
        m = re.match(r'([A-Z]+)(\d+)', c.get('r')); v = c.find(M + 'v')
        if v is not None: cells.setdefault(int(m.group(2)), {})[col(m.group(1))] = ss[int(v.text)] if c.get('t') == 's' else v.text
    return cells
cells = read_sheet(XLSX, SHEET); sheet_rec = []; cur = None
for r in sorted(cells):
    row = cells[r]; state = row.get(3, '').strip().lower()
    if 1 in row and re.fullmatch(r'\d{8}', row[1]) and 2 in row: cur = dict(date=row[1], name=row[2], states={}); sheet_rec.append(cur)      # the first row of a recording
    if cur is not None and state in ('norm', 'high', 'low') and 4 in row:
        m = re.fullmatch(r'sweep(\d+)-(\d+)', row[4].strip()); cur['states'][state] = (row[4].strip(), (int(m.group(1)) - 1) * 0.5, int(m.group(2)) * 0.5)      # sweeps -> minutes: from (first - 1) * 30 s to last * 30 s
WIN_REC = {}; rows = []
for x in sheet_rec:
    f = _match(x['name'], all_folders) or [q for q in all_folders if q.endswith(x['date']) and q.split('-')[0] == x['name'].split('-')[0]]; assert len(f) == 1, (x['name'], f)
    n_state = 'norm' if 'norm' in x['states'] else 'high'; assert n_state in x['states'] and 'low' in x['states'], x       # normoxia = norm, or high (hyperoxia, treated as normoxia); hypoxia = low
    WIN_REC[f[0]] = {'normoxia': x['states'][n_state][1:], 'hypoxia': x['states']['low'][1:]}
    rows.append(dict(recording=f[0], group=GROUP_OF.get(f[0]), normoxia_state=n_state, normoxia_sweeps=x['states'][n_state][0], normoxia_min=f'{x["states"][n_state][1]:g}-{x["states"][n_state][2]:g}', hypoxia_sweeps=x['states']['low'][0], hypoxia_min=f'{x["states"]["low"][1]:g}-{x["states"]["low"][2]:g}'))
WT = pd.DataFrame(rows).set_index('recording'); display(WT)
print(len(WT), 'recordings in the sheet;', int(((WT.normoxia_min == '0-20') & (WT.hypoxia_min == '25-40')).sum()), 'with 0-20 min normoxia and 25-40 min hypoxia; the others:'); print(WT[~((WT.normoxia_min == '0-20') & (WT.hypoxia_min == '25-40'))][['group', 'normoxia_state', 'normoxia_min', 'hypoxia_min']].to_string())
assert all(r in WIN_REC for r in USED), [r for r in USED if r not in WIN_REC]
