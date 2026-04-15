# Hypoxia dynamics

## Raw data processing
* **cell_locs.py** : copy registered cell locations to the reference brain outputs
* **locs_mat.py** : extract ephys-to-imaging frame timestamps
* **motor_ds.py** : downsample swim/ephys activity to imaging frames
* **seg_data.py** : export segmented fluorescence data and cell centers
* **datalist.csv** : primary imaging dataset index used by the segmentation export workflow
* Data assets are distributed separately through the data repository and are not documented file-by-file here.

## Baseline dynamics
### Codes
* **baseline_corr** : single-cell neural dynamics correlation with the population mean
* **baseline_ave** : population mean of positive and negative cells
* **baseline_clusters** : neural population clustering
* **baseline_subclusters** : neural subpopulation clustering within neural populations
### Relevant figures in the paper


## dFF dynamics
### Codes
* **dFF_cluster** : subcluster dFF data
* **src/dFF_dynamics/calcium_swim_fit.py** : reusable penalized lag-kernel fits for dF/F dynamics from swimming, including kernel estimation, state-wise fitting, and runtime benchmarking

### Relevant figures in the paper


## Brain maps across fish
### Codes
* **reference_brain** : edit the reference brain from confocal imaging
* **registration** : register functional imaging to the reference brain
* **reg_points** : register cells to the reference brain
### Relevant figures in the paper
* Figure 1o-p
* ED Figure 2b, c, e
* ED Figure 6b-g
* ED Figure 8b, f, h, k, m

## Behavioral data
### Codes
* **behavioral_model_fit.py** : run GLM model fits for behavioral models using O2 history only, O2 history + swim state, swim history only, and O2 history + swim history; emit `behavioral_model_fit_runs.json`; and optionally generate plots and statistics reports for pseudo-R2, AIC, and two-way ANOVA
* **behavioral_model_fit_runs.json** : per-run fit outputs with r2, AIC, params, and perfect-separation flag (ignored by git)
* **behavior_fit_report.py** : reusable plotting and two-way ANOVA reporting helpers in `src/behaviors`
### Relevant figures in the paper
* Figure 1i-l : swim action prediction from O2 history and swim history

## Notebook sharing
Notebook files may contain private metadata or outputs, including absolute filesystem paths, usernames, execution state, and other machine-specific information. Please report any such information you find in the notebooks to the maintainers so that it can be removed.

## License
This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
