# Hypoxia_dynamics

## Raw data processing
* **cell_locs.py** : copy registered cell locations to the reference brain outputs
* **locs_mat.py** : extract ephys-to-imaging frame timestamps
* **motor_ds.py** : downsample swim/ephys activity to imaging frames
* **seg_data.py** : export segmented fluorescence data and cell centers
* **datalist.csv** : primary imaging dataset index used by the segmentation export workflow
* **datalist_gfap_gc6f.csv** : GFAP GCaMP6f dataset index
* **datalist_gfap_gc6f_v2.csv** : GFAP GCaMP6f dataset index with stimulus-condition metadata
* **datalist_grabNE.csv** : GRABNE dataset index
* **datalist_huc_ablation.csv** : HuC ablation dataset index for registered cell-center copying
* **datalist_huc_h2b_gc7f.csv** : HuC H2B-GCaMP7f dataset index
* **datalist_huc_nodose.csv** : HuC nodose dataset index
* **datalist_th1_gc6f.csv** : TH1 GCaMP6f dataset index with stimulus-condition metadata
* **O2_internal.npz** : processed internal oxygen-trace bundle
* **O2_internal_20220806.mat** : MATLAB source file for internal oxygen traces
* **data.mat** : shared MATLAB data export used by the analysis notebooks
* **voltage_hypoxia.csv** : hypoxia voltage trace table

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
