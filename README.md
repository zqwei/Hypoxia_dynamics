# Hypoxia dynamics

## Raw data processing
* **cell_locs.py** : copy registered cell locations to the reference brain outputs
* **locs_mat.py** : extract ephys-to-imaging frame timestamps
* **motor_ds.py** : downsample swim/ephys activity to imaging frames
* **seg_data.py** : export segmented fluorescence data and cell centers
* **datalist.csv** : primary imaging dataset index used by the segmentation export workflow
* Data assets are distributed separately through the data repository and are not documented file-by-file here.

## Baseline dynamics
### Relevant figures in the paper


### Codes
* **baseline_corr** : single neural dynamics correlation with population mean
* **baseline_ave** : population mean of postive and negative cells
* **baseline_clusters** : neural population clustering
* **baseline_subclusters** : neural subpopulation clustering on neural populations

## dFF dynamics
### Relevant figures in the paper

### Codes
* **dFF_cluster** : subcluster dFF data


## Brain maps across fish
### Relevant figures in the paper
* Figure 1o-p
* ED Figure 2b, c, e
* ED Figure 6b-g
* ED Figure 8b, f, h, k, m

### Codes
* **reference_brain** : edit reference brain from cofocal imaging
* **registration** : registration code from functional imaging to reference
* **reg_points** : register cells to reference

## Behavioral data
### Relevant figures in the paper
* Figure 1i-l : swim action prediction from O2 history and swim history

### Codes
* **behavioral_model_fit.py** : run GLM model fits for behavioral models of O2 history only, O2 history + swim state, swim history only and O2 history + swim history, emit `behavioral_model_fit_runs.json`, and optionally generate plots (bar plots)/statistics reports (performance metrics like pseduo r2 and AIC and two-way ANOVA) for the model fits
* **behavioral_model_fit_runs.json** : per-run fit outputs with r2, AIC, params, and perfect-separation flag (ignored by git)
* **behavior_fit_report.py** : reusable plotting and two-way ANOVA reporting helpers in `src/behaviors`

## Notebook sharing
Notebook files may contain private metadata or outputs, including absolute filesystem paths, usernames, execution state, and other machine-specific information. Review and sanitize notebooks before sharing them outside the lab or publishing them publicly.

## License
This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
