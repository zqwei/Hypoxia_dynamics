import numpy as np
import os
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
df = pd.read_csv('../data/datalist.csv')
valid_list = [df.iloc[n]['registration_root']!='None' for n in range(len(df))]
df = df[valid_list]

def read_reg_mat(file):
    with open(file, 'r') as f:
        l = [[float(num) for num in line.replace(' \n', '').split(' ')] for line in f]
    return np.array(l)


for ind, row in df.iterrows():
    fimg_dir = row['save_root']
    reg_root = row['registration_root']
    if os.path.exists(fimg_dir + 'cell_center_affine_registered.npy'):
        continue

    _ = np.load(reg_root+'/sample_parameters.npz', allow_pickle=True)
    fix_range = _['fix_range']
    atlas_range = _['atlas_range']
    fix_vox = _['fix_vox']
    atlas_vox = _['atlas_vox']

    x_, y_, z_ = atlas_range
    saltas_ = sitk.Image((x_[1]-x_[0]).item(), (y_[1]-y_[0]).item(), \
                         (z_[1]-z_[0]).item(), sitk.sitkUInt8)  
    x_, y_, z_ = fix_range
    sfix_ = sitk.Image((x_[1]-x_[0]).item(), (y_[1]-y_[0]).item(), \
                       (z_[1]-z_[0]).item(), sitk.sitkUInt8)
    sfix_.SetSpacing(fix_vox[:-1])
    saltas_.SetSpacing(atlas_vox[:-1])
    sfix_.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    saltas_.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    # transforms (affines)
    affine_output = reg_root + '/atlas_fix_affine.mat'
    affine_mat = read_reg_mat(affine_output)
    
    dims = 3
    tx_affine = sitk.AffineTransform(dims)
    tx_affine.SetMatrix(affine_mat[:3, :3].reshape(-1))
    tx_affine.SetTranslation(affine_mat[:3, 3])
    tx_affine_inv = tx_affine.GetInverse()
    
    def tx_inv_low_high(point_):
        _ = sfix_.TransformContinuousIndexToPhysicalPoint(point_)
        _ = tx_affine.TransformPoint(_)
        return saltas_.TransformPhysicalPointToContinuousIndex(_)
    
    cell_centers = np.load(fimg_dir + 'cell_center.npy')
    z, x, y = cell_centers.T
    point_list_low = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T.astype('float')
    paltas_low = [tx_inv_low_high(_) for _ in point_list_low]
    paltas_low = np.array(paltas_low)
    x, y, z = paltas_low.T
    cell_centers_ = np.vstack([z.flatten(), x.flatten(), y.flatten()]).T.astype('float')
    np.save(fimg_dir + 'cell_center_affine_registered.npy', cell_centers_)