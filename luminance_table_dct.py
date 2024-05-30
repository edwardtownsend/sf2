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

def quant1_jpeg(X, step_table, N=8, rise1=None):
    if X.shape[0] != X.shape[1]:
        raise ValueError('Input array is not square!')
    if X.shape[0] % N != 0:
        raise ValueError('Need dimensions of input array to be an integer multiple of {N} for {N}x{N} DCT')

    m = X.shape[0]
    q = np.zeros(X.shape)

    for i in range(m):
        for j in range(m):
            if rise1 is None:
                rise = step_table[i % N, j % N]
                rise /= 2
            else:
                rise = rise1
            q[i, j] = np.ceil((np.abs(X[i, j]) - rise)/step_table[i % N, j % N])
    
    return q


def quant2_jpeg(q, step_table, N=8, rise1=None):
    m = q.shape[0]
    Xq = np.zeros(q.shape)

    if rise1 is None:
        for i in range(m):
            for j in range(m):
                Xq[i, j] = q[i, j] * step_table[i % N, j % N]

    else:
        # Reconstruct quantised values and incorporate sign(q).
        for i in range(m):
            for j in range(m):
                Xq = q[i, j] * step_table[i % N, j % N] + np.sign(q[i, j]) * (rise1 - step_table[i % N, j % N]/2.0)
    
    return Xq

def quantise_jpeg(x, step_table, N=8, rise1=None):
    # Perform both quantisation steps
    step_table = np.array(step_table)
    y = quant2_jpeg(quant1_jpeg(x, step_table, N, rise1), step_table, N, rise1)
    return y

def gen_Y_quant_dct_jpeg(X, step_table, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
    """
    Note C is the DCT matrix of coefficients, generated using dct_ii(N).
    C is input paramter instead of block_size to avoid having to re-compute C in every function when one function calls another.
    When s = None we perform the DCT instead of the LBT.
    """
    if s == None:
        Xp = X
    else:
        N = C.shape[0]
        Pf, Pr = pot_ii(N, s)
        t = np.s_[N//2:-N//2]
        Xp = X.copy()
        Xp[t, :] = colxfm(Xp[t, :], Pf)
        Xp[:, t] = colxfm(Xp[:, t].T, Pf).T

    Y = colxfm(colxfm(Xp, C).T, C).T
    Y = suppress_components(Y, C.shape[0], supp_comp_num)
    Yq = quantise_jpeg(Y, step_table)

    return Yq

def gen_Z_quant_dct_jpeg(X, step_table, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
    """
    Note C is the DCT matrix of coefficients, generated using dct_ii(N).
    C is input paramter instead of block_size to avoid having to re-compute C in every function when one function calls another.
    When s = None we perform the DCT instead of the LBT.
    """
    
    if s == None:
        Xp = X
    else:
        N = C.shape[0]
        Pf, Pr = pot_ii(N, s)
        t = np.s_[N//2:-N//2] 
        Xp = X.copy()
        Xp[t, :] = colxfm(Xp[t, :], Pf)
        Xp[:, t] = colxfm(Xp[:, t].T, Pf).T

    Y = colxfm(colxfm(Xp, C).T, C).T
    Y = suppress_components(Y, C.shape[0], supp_comp_num)
    Yq = quantise_jpeg(Y, step_table)
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    
    if s == None:
        return Z
    else:
        Zp = Z.copy()
        Zp[:, t] = colxfm(Zp[:, t].T, Pr.T).T
        Zp[t, :] = colxfm(Zp[t, :], Pr.T)
        return Zp

def compute_err_dct_2(X, step_size, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
    """
    Note C is the DCT matrix of coefficients, generated using dct_ii(N).
    C is input paramter instead of block_size to avoid having to re-compute C in every function when one function calls another.
    When s = None we perform the DCT instead of the LBT.
    """
    Zp = gen_Z_quant_dct_2(X, step_size, C, s, rise1_ratio, supp_comp_num)
    return np.std(X-Zp)


def find_step_ratio_equal_rms_dct(X, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
    """
    Note C is the DCT matrix of coefficients, generated using dct_ii(N).
    C is input paramter instead of block_size to avoid having to re-compute C in every function when one function calls another.
    When s = None we perform the DCT instead of the LBT.
    """
    target_err = np.std(X - quantise(X, 17))

    # Binary search
    low, high = 15, 30
    while high - low > 0.1:
        mid = (low + high) / 2
        err = compute_err_dct_2(X, mid, C, s, rise1_ratio, supp_comp_num)

        if err < target_err:
            low = mid
        else:
            high = mid

    return (low + high) / 2    