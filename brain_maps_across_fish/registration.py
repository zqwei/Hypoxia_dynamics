import numpy as np
import matplotlib.pyplot as plt
import os
from fish_proc.utils import dask_ as fdask
from fish_proc.wholeBrainDask.cellProcessing_single_WS import *
from glob import glob
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import pandas as pd


def read_h5(filename, dset_name='default'):
    import h5py
    with h5py.File(filename, 'r') as hf:
        return hf[dset_name][()]

    
def getCameraInfo(file):
    from xml.dom import minidom
    camera_info = dict()
    xmldoc = minidom.parse(file)
    itemlist = xmldoc.getElementsByTagName('info')
    for s in itemlist:
        camera_info.update(dict(s.attributes.items()))
    itemlist = xmldoc.getElementsByTagName('action')
    for s in itemlist:
        camera_info.update(dict(s.attributes.items()))
    return camera_info


def read_reg_mat(file):
    with open(file, 'r') as f:
        l = [[float(num) for num in line.replace(' \n', '').split(' ')] for line in f]
    return np.array(l)


atlas_path = r'/groups/ahrens/ahrenslab/jing/zebrafish_atlas/yumu_confocal/20150519/im/cy14_1p_stitched.h5'
atlas = np.swapaxes(read_h5(atlas_path, dset_name='channel0'),1,2).astype('float64')[::-1]
fatlas_smooth = atlas.copy()
thresh_u = np.percentile(fatlas_smooth, 99)
thresh_l = np.percentile(fatlas_smooth, 10)
fatlas_smooth[fatlas_smooth>thresh_u] = thresh_u
fatlas_smooth = fatlas_smooth - thresh_l
fatlas_smooth[fatlas_smooth<0] = 0
fatlas_smooth = fatlas_smooth/fatlas_smooth.max()

# camera file for functional imaging
moving_root = '/nrs/ahrens/jing/statemod/statedep/elavl3-GC_gfap-jRG/NG_vs_NGGU/20211216/fish02/6dpf_elavl3-GC8F-gfap-jrgeco_NGtrunc12-vs-NGGU_fish02_exp03_20211216_224637/ephys'
# functional imaging location
fimg_dir = '/nrs/ahrens/Ziqiang/Jing_Glia_project/Processed_data/20211216/fish02/6dpf_elavl3-GC8F-gfap-jrgeco_NGtrunc12-vs-NGGU_fish02_exp03_20211216_224637/im_CM0_voluseg/'
# high res imaging location
write_path = '/nrs/ahrens/jing/statemod/statedep/elavl3-GC_gfap-jRG/NG_vs_NGGU/20211216/fish02/6dpf_elavl3-GC8F-gfap-jrgeco_ref-stack_fish02_exp02_20211216_224437/processed/im_CM0_fixed_reference_sliced.h5'
# camera file for high res imaging
ind_ = write_path.find('/processed/')
fixed_root = write_path[:ind_+1]
# extracting fish name for temporary folder
fish_name = fimg_dir.split('/')[-3]+'_im_CM0'

fimg = np.load(fimg_dir + 'Y_ave.npy').squeeze().astype('float')
fix = read_h5(write_path).astype('float')

fimg_smooth = fimg.copy()
thresh_u = np.percentile(fimg, 99)
thresh_l = np.percentile(fimg, 75)
fimg_smooth[fimg_smooth>thresh_u] = thresh_u
fimg_smooth = fimg_smooth - thresh_l
fimg_smooth[fimg_smooth<0] = 0
fimg_smooth = fimg_smooth/fimg_smooth.max()
fimg = None

ffix_smooth = fix.copy()
thresh_u = np.percentile(ffix_smooth, 99)
thresh_l = np.percentile(ffix_smooth, 10)
ffix_smooth[ffix_smooth>thresh_u] = thresh_u
ffix_smooth = ffix_smooth - thresh_l
ffix_smooth[ffix_smooth<0] = 0
ffix_smooth = ffix_smooth/ffix_smooth.max()
fix = None

cinfo = getCameraInfo(moving_root[:-6] + '/im_CM0/ch0.xml')
fimg_zvox = float(cinfo['z_step'])

cinfo = getCameraInfo(fixed_root + '/im_CM0/ch0.xml')
fix_zvox = float(cinfo['z_step'])

atlas_zvox = 2.
spim_res = 0.406
spim_res_ = 0.406*2
fimg_vox = np.array([spim_res_, spim_res_, fimg_zvox, 1])
fix_vox = np.array([spim_res, spim_res, fix_zvox, 1])
atlas_vox = np.array([spim_res, spim_res, atlas_zvox, 1])

print(fimg_vox, fix_vox, atlas_vox)
print(ffix_smooth.shape, fimg_smooth.shape)

x_sliced = ffix_smooth.shape[-1]
fig, ax = plt.subplots(1, 3, figsize=(20, 4))
ax[0].imshow(ffix_smooth[:, :,::-1].max(0), cmap='Greens', aspect='auto')
ax[1].imshow(fimg_smooth[:,:,::-1].max(0), cmap='Reds', aspect='auto')
ax[2].imshow(fatlas_smooth[:170, :, :].max(0), cmap='Blues', aspect='auto')
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(20, 4))
ax[0].imshow(ffix_smooth[:, :,::-1].max(1), cmap='Greens', aspect='auto')
ax[1].imshow(fimg_smooth[:,:,::-1].max(1), cmap='Reds', aspect='auto')
ax[2].imshow(fatlas_smooth[:170, :, :].max(1), cmap='Blues', aspect='auto')
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(20, 4))
ax[0].imshow(ffix_smooth[:, :,::-1].max(2), cmap='Greens', aspect='auto')
ax[1].imshow(fimg_smooth[:,:,::-1].max(2), cmap='Reds', aspect='auto')
ax[2].imshow(fatlas_smooth[:170, :, :].max(2), cmap='Blues', aspect='auto')
plt.show()

fimg_smooth_ = fimg_smooth[:,:,::-1]
ffix_smooth_ = ffix_smooth[:,:,::-1]
fatlas_smooth_ = fatlas_smooth[:170, :, :]
altas_ = nib.Nifti1Image(fatlas_smooth_.swapaxes(0, 2), affine=np.diag(atlas_vox))
fix_ = nib.Nifti1Image(ffix_smooth_.swapaxes(0, 2), affine=np.diag(fix_vox))
fimg_ = nib.Nifti1Image(fimg_smooth_.swapaxes(0, 2), affine=np.diag(fimg_vox))


# temporal file location for processing (change if using it for different locations)
pre_fix_root = '/nrs/ahrens/Ziqiang/scratch/registration/'+fish_name + '/'


if not os.path.exists(pre_fix_root):
    os.makedirs(pre_fix_root)

nib.save(altas_, pre_fix_root+'altas.nii.gz')
nib.save(fix_, pre_fix_root+'fix.nii.gz')
nib.save(fimg_, pre_fix_root+'fimg.nii.gz')

fz, fy, fx = ffix_smooth_.shape
az, ay, ax = fatlas_smooth_.shape
print((fz, fy, fx), (az, ay, ax))
np.savez(pre_fix_root+'sample_parameters', fix_range=np.array([[ 0, fx],[0, fy], [0, fz]]), \
                                        atlas_range=np.array([[0, ax], [ 0, ay], [0, 170]]), \
                                        fimg_vox = fimg_vox, fix_vox = fix_vox, atlas_vox = atlas_vox, \
                                        flip_xyz = np.array([1, 0, 0]).astype('bool')) # [1, 0, 0]
print(pre_fix_root)


### affine transform from functional imaging to high-res one
altas_file = pre_fix_root+'altas.nii.gz'
fix_file = pre_fix_root+'fix.nii.gz'
fimg_file = pre_fix_root+'fimg.nii.gz'

# fimg_file to fix_file
output = pre_fix_root + 'fimg_fix_affine.mat'
greedy = 'greedy -d 3 -dof 6 -a -m MI -i ' + fix_file + ' ' + fimg_file + ' -o ' + output + ' -ia-image-centers -n 200x100x50'
os.system(greedy)

fimg_fix_affine_align = pre_fix_root+'fimg_fix_affine_aglin.nii.gz'
greedy_reg = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + fimg_file + ' ' + fimg_fix_affine_align + ' -ri LINEAR -r ' + output
os.system(greedy_reg)

fimg_reg = nib.load(fimg_fix_affine_align)

plt.imshow(fix_.get_data().max(0), cmap='Greens')
plt.imshow(fimg_reg.get_data().max(0), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(1), cmap='Greens')
plt.imshow(fimg_reg.get_data().max(1), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(2), cmap='Greens')
plt.imshow(fimg_reg.get_data().max(2), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()


## affine transform from atlas imaging to high-res one
### rigid for quick center location alignments

## affine
output = pre_fix_root + 'atlas_fix_rigid.mat'
greedy = 'greedy -d 3 -dof 6 -a -m MI -i ' + fix_file + ' ' + altas_file + ' -o ' + output + ' -ia-image-centers -n 200x100x50'
os.system(greedy)

output = pre_fix_root + 'atlas_fix_rigid.mat'
atlas_fix_rigid_align = pre_fix_root + 'atlas_fix_rigid.nii.gz'
greedy_reg = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + altas_file + ' ' + atlas_fix_rigid_align + ' -ri LINEAR -r ' + output
os.system(greedy_reg)

atlas_rigid_reg = nib.load(atlas_fix_rigid_align)
plt.imshow(fix_.get_data().max(2), cmap='Greens')
plt.imshow(atlas_rigid_reg.get_data().max(2), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(0), cmap='Greens')
plt.imshow(atlas_rigid_reg.get_data().max(0), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(1), cmap='Greens')
plt.imshow(atlas_rigid_reg.get_data().max(1), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

### Affine transform for simple scaling
## affine
rigid_output = pre_fix_root + 'atlas_fix_rigid.mat'
output = pre_fix_root + 'atlas_fix_affine.mat'
greedy = 'greedy -d 3 -dof 12 -a -m NMI -i ' + fix_file + ' ' + altas_file + ' -o ' + output + ' -ia ' + rigid_output + ' -n 50x20x10'
os.system(greedy)

atlas_fix_affine_align = pre_fix_root + 'atlas_fix_affine.nii.gz'
output = pre_fix_root + 'atlas_fix_affine.mat'
greedy_reg = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + altas_file + ' ' + atlas_fix_affine_align + ' -ri LINEAR -r ' + output
os.system(greedy_reg)

atlas_affine_reg = nib.load(atlas_fix_affine_align)
plt.imshow(fix_.get_data().max(2), cmap='Greens')
plt.imshow(atlas_affine_reg.get_data().max(2), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(0), cmap='Greens')
plt.imshow(atlas_affine_reg.get_data().max(0), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(1), cmap='Greens')
plt.imshow(atlas_affine_reg.get_data().max(1), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

### Deformable registration
## affine
affine_output = pre_fix_root + 'atlas_fix_affine.mat'
output = pre_fix_root + 'atlas_fix_wrap_mat.nii.gz'
output_inv = pre_fix_root + 'atlas_fix_wrap_inv_mat.nii.gz'
smoothing = '1.7vox 0.7vox' # Deformable registration smoothing parameters
greedy = 'greedy -d 3 -m NCC 12x12x6 -i ' + fix_file + ' ' + altas_file + ' -it ' + affine_output + ' -o ' + output + ' -n 100x40x20 -s ' + smoothing + ' -oinv ' + output_inv
os.system(greedy)

atlas_fix_wrap_align = pre_fix_root + 'atlas_fix_wrap.nii.gz'
greedy = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + altas_file + ' ' + atlas_fix_wrap_align + ' -ri LINEAR -r ' + output + ' ' + affine_output
os.system(greedy)
# #### invert
atlas_fix_affine_wrap_inv = pre_fix_root + 'atlas_fix_wrap_inv.nii.gz'
greedy = 'greedy -d 3 -rf ' + altas_file + ' -rm ' + fix_file + ' ' + atlas_fix_affine_wrap_inv + ' -r ' + affine_output + ',-1 ' + output_inv
os.system(greedy)

atlas_wrap_reg = nib.load(atlas_fix_wrap_align)
plt.imshow(fix_.get_data().max(2), cmap='Greens')
plt.imshow(atlas_wrap_reg.get_data().max(2), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(0), cmap='Greens')
plt.imshow(atlas_wrap_reg.get_data().max(0), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(1), cmap='Greens')
plt.imshow(atlas_wrap_reg.get_data().max(1), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()
