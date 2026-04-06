import numpy as np


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


def cell_loc(X, Y, Z, W, cell_id):
    thres_ = 0.005
    w_ = (W[cell_id] > thres_) * W[cell_id]
    w_[np.isnan(w_)] = 0
    x_loc, y_loc, z_loc = X[cell_id], Y[cell_id], Z[cell_id]
    return (z_loc.dot(w_)) / w_.sum(), (x_loc.dot(w_)) / w_.sum(), (y_loc.dot(w_)) / w_.sum()
