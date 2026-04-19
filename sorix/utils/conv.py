import numpy as np
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp
else:
    cp = None

_im2col_cache = {}

def get_im2col_indices(x_shape, field_height, field_width, padding=(1, 1), stride=(1, 1)):
    # First figure out output sizes
    N, C, H, W = x_shape
    
    # Indices k, i, j only depend on C, H, W, field sizes, padding and stride, not N
    key = (C, H, W, field_height, field_width, padding, stride)
    if key in _im2col_cache:
        return _im2col_cache[key]
        
    ph, pw = padding
    sh, sw = stride
    out_height = int((H + 2 * ph - field_height) // sh + 1)
    out_width = int((W + 2 * pw - field_width) // sw + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = sh * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = sw * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    result = (k.astype(int), i.astype(int), j.astype(int))
    _im2col_cache[key] = result
    return result


def im2col_indices(x, field_height, field_width, padding=(1, 1), stride=(1, 1)):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    xp = cp if (cp is not None and isinstance(x, cp.ndarray)) else np
    ph, pw = padding
    # Ensure ph, pw are standard ints to avoid issues with some libraries (like coverage) 
    # and use a list of tuples for pad_width.
    pad_width = [(0, 0), (0, 0), (int(ph), int(ph)), (int(pw), int(pw))]
    x_padded = xp.pad(x, pad_width, mode='constant', constant_values=0)

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=(1, 1),
                   stride=(1, 1)):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    xp = cp if (cp is not None and isinstance(cols, cp.ndarray)) else np
    ph, pw = padding
    H_padded, W_padded = H + 2 * ph, W + 2 * pw
    x_padded = xp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    
    if xp is np:
        # NumPy optimization: np.add.at is extremely slow on CPU.
        # We use bincount instead to vastly speed up the accumulation.
        flat_indices = np.ravel_multi_index((k, i, j), (C, H_padded, W_padded)).ravel()
        target_flat_size = C * H_padded * W_padded
        for batch_idx in range(N):
            res = np.bincount(flat_indices, weights=cols_reshaped[batch_idx].ravel(), minlength=target_flat_size)
            x_padded[batch_idx] = res.reshape(C, H_padded, W_padded)
    else:
        # Use add.at to accumulate gradients correctly (handles overlapping windows)
        xp.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if ph == 0 and pw == 0:
        return x_padded
    
    # Correct slicing for non-symmetric padding if needed, but here padding is ph, pw
    return x_padded[:, :, ph:H_padded-ph, pw:W_padded-pw]
