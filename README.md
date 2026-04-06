# Hypoxia_dynamics

## Raw data processing
* **cell_locs.py** : copy registered cell locations to the reference brain outputs
* **locs_mat.py** : extract ephys-to-imaging frame timestamps
* **motor_ds.py** : downsample swim/ephys activity to imaging frames
* **seg_data.py** : export segmented fluorescence data and cell centers
* **datalist.csv** : primary imaging dataset index used by the segmentation export workflow
* Data assets are distributed separately through the data repository and are not documented file-by-file here.

## Baseline dynamics
* **baseline_corr** : single neural dynamics correlation with population mean
* **baseline_ave** : population mean of postive and negative cells
* **baseline_clusters** : neural population clustering
* **baseline_subclusters** : neural subpopulation clustering on neural populations

## dFF dynamics
* **dFF_cluster** : subcluster dFF data


## Brain maps across fish
* **reference_brain** : edit reference brain from cofocal imaging
* **registration** : registration code from functional imaging to reference
* **reg_points** : register cells to reference
