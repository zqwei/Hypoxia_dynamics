import numpy as np
from h5py import File
import dask.array as da


def baseline(data, window=100, percentile=15, downsample=10, axis=-1):
    from scipy.ndimage import percentile_filter
    from scipy.interpolate import interp1d
    from numpy import ones
    size = ones(data.ndim, dtype='int')
    size[axis] *= window // downsample
    if downsample == 1:
        bl = percentile_filter(data, percentile=percentile, size=size)
    else:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, None, downsample)
        data_ds = data[tuple(slices)].astype('float')
        baseline_ds = percentile_filter(data_ds, percentile=percentile, size=size)
        interper = interp1d(range(0, data.shape[axis], downsample), baseline_ds,
                            axis=axis, fill_value='extrapolate')
        bl = interper(range(data.shape[axis]))
    return bl.astype(data.dtype)
import os
import pandas as pd
# df = pd.read_csv('datalist_huc_nodose.csv', index_col=0)
df = pd.read_csv('datalist.csv', index_col=0)

def cell_loc(cell_id):
    thres_ = 0.005
    w_ = (W[cell_id]>thres_)*W[cell_id]
    w_[np.isnan(w_)]=0
    x_loc, y_loc, z_loc = X[cell_id],Y[cell_id],Z[cell_id]
    return (z_loc.dot(w_))/w_.sum(), (x_loc.dot(w_))/w_.sum(), (y_loc.dot(w_))/w_.sum()


for ind, row in df.iterrows():
    if ind<38:
        continue
    dir_ = row['dir_']+'/seg_mika/'
    if not os.path.exists(dir_):
        continue
    save_root = row['save_root']
    if os.path.exists(save_root+'cell_center.npy'):
        continue
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print(f'Processing {ind} at {save_root}')
    cell_file = File(dir_+'cells0_clean.hdf5', 'r')
    volume_file = File(dir_+'volume0.hdf5', 'r')

    X=cell_file['cell_x']
    Y=cell_file['cell_y']
    Z=cell_file['cell_z']
    W=cell_file['cell_weights'][()]
    V=cell_file['volume_weight']
    
    F = cell_file['cell_timeseries_raw'][()]
    background = cell_file['background'][()]
    brain_map=volume_file['volume_mean'][()]
    
    # remove background from raw F
    F = F - (background - 10)
    F[F<0]=0
    F_dask = da.from_array(F, chunks=('auto', -1))
    win_ = 400
    baseline_ = da.map_blocks(baseline, F_dask, dtype='float', window=win_, percentile=20, downsample=10).compute()
    dFF = F/baseline_-1

    brain_shape = V.shape
    np.savez(save_root+'cell_dff.npz', \
             dFF=dFF.astype('float16'), \
             baseline=baseline_.astype('float16'), \
             brain_shape=brain_shape, \
             X=X, Y=Y, Z=Z, W=W)
    np.save(save_root+'Y_ave.npy', brain_map)
    
    numCells = F.shape[0]
    A_center = np.zeros((numCells,3))
    for n_cell in range(numCells):
        A_center[n_cell] = cell_loc(n_cell)
    np.save(save_root+'cell_center.npy', A_center)

