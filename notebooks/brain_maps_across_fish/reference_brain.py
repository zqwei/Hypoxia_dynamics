import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from czifile import imread
reference_ = '/nrs/ahrens/Rongwei/data_to_Ziqiang/confocal/fish01-6dpf-huc-h2b-gc7f-vglut2a-DsRed-zstack.czi'
ref_ = imread(reference_).squeeze()
ref_ = ref_[:, :, :, ::-1]
np.save('/nrs/ahrens/Ziqiang/Atlas/atlas_RZ/atlas.npy', ref_)

atlas = np.load('/nrs/ahrens/Ziqiang/Atlas/atlas_RZ/atlas.npy')

fatlas_smooth = atlas[0].copy()
thresh_u = np.percentile(fatlas_smooth, 99)
thresh_l = np.percentile(fatlas_smooth, 10)
fatlas_smooth[fatlas_smooth>thresh_u] = thresh_u
fatlas_smooth = fatlas_smooth - thresh_l
fatlas_smooth[fatlas_smooth<0] = 0
fatlas_smooth = fatlas_smooth/fatlas_smooth.max()
atlas_vox = np.array([0.518, 0.518, 1, 1])
altas_ = nib.Nifti1Image(fatlas_smooth.swapaxes(0, 2), affine=np.diag(atlas_vox))
nib.save(altas_, '/nrs/ahrens/Ziqiang/Atlas/atlas_RZ/altas_h2b_gcamp7f.nii.gz')

fatlas_smooth = atlas[1].copy()
thresh_u = np.percentile(fatlas_smooth, 99)
thresh_l = np.percentile(fatlas_smooth, 10)
fatlas_smooth[fatlas_smooth>thresh_u] = thresh_u
fatlas_smooth = fatlas_smooth - thresh_l
fatlas_smooth[fatlas_smooth<0] = 0
fatlas_smooth = fatlas_smooth/fatlas_smooth.max()
atlas_vox = np.array([0.518, 0.518, 1, 1])
altas_ = nib.Nifti1Image(fatlas_smooth.swapaxes(0, 2), affine=np.diag(atlas_vox))
nib.save(altas_, '/nrs/ahrens/Ziqiang/Atlas/atlas_RZ/altas_vglut2a_DsRed.nii.gz')