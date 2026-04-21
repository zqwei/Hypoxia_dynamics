# Hypoxia dynamics

## Raw data processing
* **cell_locs.py** : copy registered cell locations into the reference-brain outputs
* **locs_mat.py** : extract electrophysiology-to-imaging frame timestamps
* **motor_ds.py** : downsample swim/ephys activity to imaging frames
* **seg_data.py** : export segmented fluorescence traces and cell centers
* **datalist.csv** : primary imaging dataset index used by the segmentation export workflow
* Data assets are distributed separately through the data repository and are not documented file-by-file here.

## Baseline dynamics
### Codes
* **baseline_corr.py** : compute single-cell baseline-dynamics correlations with the population mean
* **baseline_ave.py** : compute population means for positive and negative cells
* **baseline_clusters.py** : cluster neural populations by baseline dynamics
* **baseline_subclusters.py** : cluster neural subpopulations within the baseline-dynamics populations
* **baseline_stats.py** : summarize baseline-dynamics statistics
* **oxy_baseline_brain_map.py** : generate across-fish baseline brain maps
### Relevant figures in the paper


## dFF dynamics
### Codes
* **dFF_cluster_R1.py / dFF_cluster_R2.py / dFF_cluster_R3.py** : cluster and subcluster dF/F dynamics
* **dFF_cluster_act.py** : summarize cluster activity traces
* **GLM_calcium_swim_fit.py** : run penalized lag-kernel fits of dF/F dynamics from swimming, including kernel estimation, state-wise fitting, and runtime benchmarking
* **brain_map_beta_ratio.py** : generate brain maps of swim-calcium modulation ratios from the GLM beta estimates
* **dFF_d_prime.py** : compute d' for dF/F dynamics

### Relevant figures in the paper


## Brain maps across fish
### Codes
* **reference_brain.py** : prepare the reference brain from confocal imaging
* **registration.py** : register functional imaging volumes to the reference brain
* **reg_points_affine.py** : register cell coordinates to the reference brain
* **brain_map_neg_pos_oxy.py** : generate across-fish positive and negative oxygen-response maps
* **brain_map_motor_clamp.py** : generate across-fish motor-clamp response maps
### Relevant figures in the paper
* Figure 1o-p
* ED Figure 2b, c, e
* ED Figure 6b-g
* ED Figure 8b, f, h, k, m

## Behavioral data
### Codes
* **behavioral_model_fit.py** : run GLM behavioral models using O2 history only, O2 history plus swim state, swim history only, and O2 history plus swim history; emit `behavioral_model_fit_runs.json`; and optionally generate plots and statistics reports for pseudo-R2, AIC, and two-way ANOVA
* **behavioral_model_fit_runs.json** : per-run fit outputs with r2, AIC, params, and perfect-separation flag (ignored by git)
* **src/behaviors/behavior_fit_report.py** : reusable plotting and two-way ANOVA reporting helpers
### Relevant figures in the paper
* Figure 1i-l : swim action prediction from O2 history and swim history

## Notebook sharing
Notebook files may contain private metadata or outputs, including absolute filesystem paths, usernames, execution state, and other machine-specific information. Please report any such information you find in the notebooks to the maintainers so it can be removed before external sharing.

## License
This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
