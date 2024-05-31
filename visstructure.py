import numpy as np
from scipy.ndimage import gaussian_filter

def ssimTEST2(X, Y, win_size=None, gradient=False, data_range=None, multichannel=False, gaussian_weights=False, full=False):
    """Compute the Structural Similarity Index (SSIM) between two images.

    Parameters
    ----------
    X, Y : ndarray
        Images.  Any dimensionality with same shape.
    win_size : int or None
        The side-length of the sliding window used in comparison. Must be an odd
        value. If `gaussian_weights` is True, this is ignored and the window size
        will depend on `sigma`.
    gradient : bool
        If True, also return gradient.
    data_range : int or None
        The data range of the input image (distance between minimum and maximum
        possible values).  By default, this is estimated from the image data-type.
    multichannel : bool
        If True, treat the last dimension of the array as channels. Similarity
        calculations are done independently for each channel then averaged.
    gaussian_weights : bool
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5. If False, the patch
        means and variances are estimated from the central pixels without
        spatial weighting.
    full : bool
        If True, return the full structural similarity image instead of the
        mean value.

    Returns
    -------
    mssim : ndarray
        The mean structural similarity over the image.  If `full` is True,
        return the full SSIM image.
    grad : ndarray
        The gradient of the structural similarity index between X and Y.
        This is None if `gradient` is False.
    """

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if win_size is None:
        win_size = 11
        if gaussian_weights:
            win_size = 11  # for sigma = 1.5
    elif win_size < 2:
        raise ValueError('`win_size` must be an integer >= 2.')

    if win_size % 2 == 0:
        raise ValueError('`win_size` must be an odd integer.')

    if data_range is None:
        dmin, dmax = np.min(X), np.max(X)
        data_range = dmax - dmin

    ndim = X.ndim

    if ndim == 2:
        grad = None
        func = _ssim
    elif ndim == 3:
        if multichannel:
            # loop over channels
            ssim = []
            grad = []
            for channel in range(X.shape[-1]):
                ch_result, ch_grad = ssim(X[..., channel], Y[..., channel], win_size=win_size, gradient=True,
                                           data_range=data_range, gaussian_weights=gaussian_weights, full=True)
                ssim.append(ch_result)
                grad.append(ch_grad)
            return np.mean(ssim), np.mean(grad, axis=0)
        else:
            func = _ssim
            grad = None
    else:
        raise ValueError('Input images must have either 2 or 3 dimensions.')

    mssim = func(X, Y, win_size=win_size, gradient=gradient, data_range=data_range,
                 gaussian_weights=gaussian_weights, full=full)

    return mssim, grad

def _fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    coords = np.arange(0, size, dtype=np.float32) - (size - 1) / 2.0
    coords_x = np.tile(coords, (size, 1))
    coords_y = np.tile(coords, (size, 1)).T

    raw_filter = np.exp(-(coords_x ** 2 + coords_y ** 2) / (2 * sigma ** 2))
    kernel = raw_filter / np.sum(raw_filter)

    return kernel

def _ssim(X, Y, win_size=None, gradient=False, data_range=None, gaussian_weights=False, full=False):
    """Compute the Structural Similarity Index (SSIM) between two images."""
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')
    if win_size is None:
        win_size = 11
        if gaussian_weights:
            win_size = 11  # for sigma = 1.5
    elif win_size < 2:
        raise ValueError('`win_size` must be an integer >= 2.')
    if win_size % 2 == 0:
        raise ValueError('`win_size` must be an odd integer.')
    if data_range is None:
        dmin, dmax = np.min(X), np.max(X)
        data_range = dmax - dmin
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    if gaussian_weights:
        filter_func = _fspecial_gauss
    else:
        filter_func = np.ones((win_size, win_size)) / (win_size ** 2)
    pad = (win_size - 1) // 2
    # X and Y are flattened into vectors
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    # Compute local means
    mu_X = gaussian_filter(X, sigma)
    mu_Y = gaussian_filter(Y, sigma)
    # Compute local variances and covariances
    sigma_X2 = gaussian_filter(X ** 2, sigma) - mu_X ** 2
    sigma_Y2 = gaussian_filter(Y ** 2, sigma) - mu_Y ** 2
    sigma_XY = gaussian_filter(X * Y, sigma) - mu_X * mu_Y

    c1 = (K1 * data_range) ** 2
    c2 = (K2 * data_range) ** 2
    # Compute SSIM components
    num = (2 * mu_X * mu_Y + c1) * (2 * sigma_XY + c2)
    den = (mu_X ** 2 + mu_Y ** 2 + c1) * (sigma_X2 + sigma_Y2 + c2)
    ssim_map = num / den
    if full:
        return ssim_map
    else:
        return np.mean(ssim_map)
def visualerror(X, Z):
    Value, map = ssimTEST2(X, Z, data_range=255)

"""
def choosescheme(X):
    DWTchoice = False
    # Z1 = LBT function(X)
    # Z2 = DWT function(X)
    S1 = visualerror(X,Z1)
    S2 = visualerror(X,Z2)
    if S1> S2:
        DWTchoice = False
    elif S2> S1:
        DWTchoice =  True
    else:
        DWTchoice =  True
    if DWTchoice  == False
    # Execute full LBT code
    if DWTchoice  == False
    # Execute full DWT code
"""