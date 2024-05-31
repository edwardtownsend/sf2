import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.dct import colxfm, dct_ii
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from cued_sf2_lab.dwt import dwt, idwt
from useful_functions import *

# New functions to apply the JPEG DCT luminance quantisation table

def gen_step_table(step_table_type):
    """
    0 = JPEG luminance table (p147)
    1 = Uniform quantisation
    """

    if step_table_type == 0:
        step_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float64)

    elif step_table_type == 1:
        step_table = np.ones((8, 8), dtype=np.float64)

    else:
        raise ValueError(f'Invalid step table: step_table_type = {step_table_type}!')

    return step_table


def quant1_jpeg(X, step_table):
    N = step_table.shape[0]
    if step_table.shape[0] != step_table.shape[1]:
        raise ValueError('Step table array is not square!')
    if X.shape[0] != X.shape[1]:
        raise ValueError('Input array is not square!')
    if X.shape[0] % N != 0:
        raise ValueError('Need dimensions of input array to be an integer multiple of {N} for {N}x{N} DCT')

    m = X.shape[0]
    q = np.zeros(X.shape)

    for i in range(m):
        for j in range(m):
            step = step_table[i % N, j % N]
            rise = step / 2
            temp = np.ceil((np.abs(X[i, j]) - rise)/step)
            q[i, j] = temp*(temp > 0)*np.sign(X[i, j])

    return q

def quant2_jpeg(q, step_table):
    N = step_table.shape[0]
    m = q.shape[0]
    Xq = np.zeros(q.shape)

    for i in range(m):
        for j in range(m):
            step = step_table[i % N, j % N]
            rise = step / 2
            Xq[i, j] = q[i, j] * step + np.sign(q[i, j]) * (rise - step/2.0)
    
    return Xq

def quantise_jpeg(X, step_table):
    # Perform both quantisation steps
    Y = quant2_jpeg(quant1_jpeg(X, step_table), step_table)
    return Y


def compute_err_dct_jpeg(X, step_table, C, rise1_ratio=0.5, supp_comp_num=0):
    """
    Note C is the DCT matrix of coefficients, generated using dct_ii(N).
    C is input paramter instead of block_size to avoid having to re-compute C in every function when one function calls another.
    When s = None we perform the DCT instead of the LBT.
    """
    Y = forward_dct_lbt(X, C, None, rise1_ratio, supp_comp_num)
    Yq = quantise_jpeg(Y, step_table)
    Zp = inverse_dct_lbt(Yq, C, None, rise1_ratio, supp_comp_num)

    return np.std(X-Zp)


def find_ssr_equal_rms_dct_jpeg(X, step_table, C, rise1_ratio=0.5, supp_comp_num=0):
    """
    Note C is the DCT matrix of coefficients, generated using dct_ii(N).
    C is input paramter instead of block_size to avoid having to re-compute C in every function when one function calls another.
    When s = None we perform the DCT instead of the LBT.
    """
    target_err = np.std(X - quantise(X, 17))

    # Binary search
    low, high = 0, 1
    while high - low > 0.05:
        mid = (low + high) / 2
        err = compute_err_dct_jpeg(X, mid*step_table, C, rise1_ratio, supp_comp_num)

        if err < target_err:
            low = mid
        else:
            high = mid

    return (low + high) / 2    