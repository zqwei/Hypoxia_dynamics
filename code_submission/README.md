# Swim probability, oxygen and swim history (code)

Models of the probability that a paralysed larval zebrafish starts a swim in a 1 s bin, as a function of the oxygen level and of its own recent swims
(normoxia and hypoxia), with cross-validation and a free-running simulation of held-out blocks.

| file | content |
|---|---|
| `swim_oxygen_models.py` | all the functions: data binning and windows, the five models and their penalised fits, cross-validation, R² / AIC, free-running simulation, transition matrices. Self-contained (numpy, scipy, pandas, h5py, joblib). |
| `01_fit_swim_oxygen_models.ipynb` | data -> windows -> fits -> held-out R² / AIC -> kernels. Writes `fits.pkl`. |
| `02_free_running_simulation.ipynb` | reads `fits.pkl`; free runs of every held-out block, transition matrices P(state(t+k) \| state(t)), errors of the models. |

## Run
1. The recordings are in the folder `figshare/` next to the notebooks (7 recordings, about 2 GB; `DATA_ROOT = 'figshare/'` in `01_...`; change it if you move the data); one folder per recording, see below.
2. Run `01_...` (about 1 min on 64 cores), then `02_...` (about 15 s). `N_JOBS` sets the number of parallel workers.

All settings are in `Settings` (`swim_oxygen_models.py`); the defaults are those of the reported fits.

## Data (one folder per recording; 7 recordings)
* `data.mat` (HDF5, group `data`): `fltCh2` and `back2` (swim signal and its baseline, 6 kHz, 1 x N), `swimStartIndT` and `swimEndIndT` (start / end sample of every detected swim).
* `O2_real.mat` (HDF5): `O2_real` (oxygen, 6 kHz, 1 x N; normalised in the code to the mean of minutes 5-10).
* Only the arrays above, the first 60 min of each recording.