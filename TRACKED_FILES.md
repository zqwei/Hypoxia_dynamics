# Tracked Files — Pre-Publication Review

Review this list before making the repository public.
Mark each file as ✅ Keep | ❌ Remove | ⚠️ Review

---

## Root

| File | Status | Notes |
|------|--------|-------|
| `README.md` | ✅ Keep | |
| `LICENSE` | ✅ Keep | |
| `.gitignore` | ✅ Keep | |
| `requirements.txt` | ✅ Keep | |
| `TRACKED_FILES.md` | ⚠️ Review | This file — remove before final public release |

---

## behavioral_data/

| File | Status | Notes |
|------|--------|-------|
| `utils.py` | ⚠️ Review | |

---

## brain_maps_across_fish/

| File | Status | Notes |
|------|--------|-------|
| `brain_map_motor_clamp.py` | ⚠️ Review | |
| `brain_map_neg_pos_oxy.py` | ⚠️ Review | |
| `reference_brain.py` | ⚠️ Review | |
| `reg_points_affine.py` | ⚠️ Review | |
| `registration.py` | ⚠️ Review | |

---

## data/

| File | Status | Notes |
|------|--------|-------|
| `cell_locs.py` | ⚠️ Review | |
| `locs_mat.py` | ⚠️ Review | |
| `motor_ds.py` | ⚠️ Review | |
| `seg_data.py` | ⚠️ Review | |

---

## models/

| File | Status | Notes |
|------|--------|-------|
| `ephys_swim.py` | ⚠️ Review | |
| `free_swim.py` | ⚠️ Review | |

---

## baseline_dynamics/

| File | Status | Notes |
|------|--------|-------|
| `baseline_ave.py` | ⚠️ Review | |
| `baseline_clusters.py` | ⚠️ Review | |
| `baseline_corr.py` | ⚠️ Review | |
| `baseline_stats.py` | ⚠️ Review | |
| `baseline_subclusters.py` | ⚠️ Review | |
| `oxy_baseline_brain_map.py` | ⚠️ Review | |
| `utils.py` | ⚠️ Review | |

---

## neural_dynamics_dFF/

| File | Status | Notes |
|------|--------|-------|
| `dFF_cluster_R1.py` | ⚠️ Review | Already committed |
| `dFF_cluster_R2.py` | ⚠️ Review | Already committed |
| `dFF_cluster_R3.py` | ⚠️ Review | Already committed |
| `dFF_cluster_R3_dynamics.py` | ⚠️ Review | Already committed |
| `cluster_d_prime_precompute.py` | ⚠️ Review | New — pending commit |
| `dFF_cluster_act.py` | ⚠️ Review | New — pending commit |
| `dFF_cluster_anm_parameters.py` | ⚠️ Review | New — pending commit |
| `dFF_state_d_prime.py` | ⚠️ Review | New — pending commit |
| `utils_cluster_anm.py` | ⚠️ Review | New — pending commit |
| `tested_models.md` | ⚠️ Review | Markdown doc — include? |

---

## NOT Tracked (excluded by .gitignore)

| Pattern | Examples |
|---------|---------|
| `*.ipynb` | All notebooks |
| `*.mat` | `data/data.mat`, `atlas_fix_rigid.mat` |
| `*.csv` | All datalist CSVs in `data/` |
| `*.pdf` | All output figures in `behavioral_data/`, `models/` |
| `*.png`, `*.tif` | All images |
| `*.npz`, `*.npy` | All model output arrays |
| `depreciated/` | All deprecated subdirectories |
