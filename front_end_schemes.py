import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.dct import colxfm, dct_ii, regroup
from cued_sf2_lab.lbt import pot_ii
from useful_functions import *

# Forward and inverse DCT/LBT, set s=None to generate DCT
def forward_dct_lbt(X, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
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

    return Y

def inverse_dct_lbt(Y, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
    Z = colxfm(colxfm(Y.T, C.T).T, C.T)
    
    if s == None:
        return Z
    else:
        N = C.shape[0]
        Pf, Pr = pot_ii(N, s)
        t = np.s_[N//2:-N//2]
        Zp = Z.copy()
        Zp[:, t] = colxfm(Zp[:, t].T, Pr.T).T
        Zp[t, :] = colxfm(Zp[t, :], Pr.T)
        return Zp

# Equal rms DCT/LBT. Returns compression ratio, Z
def gen_dct_lbt_equal_rms(X, block_size, s=None, rise1_ratio=0.5, supp_comp_num=0):
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        return "Error the block size is not a power of two!"

    C = dct_ii(block_size)
    step_size = find_step_equal_rms_dct_lbt(X, C, s, rise1_ratio, supp_comp_num)
    Y = forward_dct_lbt(X, C, s, rise1_ratio, supp_comp_num)
    Yq = quantise(Y, step_size, rise1_ratio*step_size)
    Yr = regroup(Yq, block_size)
    Yr_ent = dctbpp(Yr, 16)
    X_quant = quantise(X, 17)
    comp_ratio = entropy(X_quant) / Yr_ent
    
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    if s == None:
        return comp_ratio, Z
    else:
        Zp = Z.copy()
        Pf, Pr = pot_ii(block_size, s)
        t = np.s_[block_size//2:-block_size//2]
        Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
        Zp[t,:] = colxfm(Zp[t,:], Pr.T)
        return comp_ratio, Zp

# Forward and inverse DWT
def nlevdwt(X, n):
    Y = X.copy()
    m = X.shape[0]
    for i in range(n):
        Y[:m,:m] = dwt(Y[:m,:m])
        m //= 2
    
    return Y

def nlevidwt(Y, n):
    m = Y.shape[0] // (2 ** (n - 1))
    Z = Y.copy()
    for i in range(n):
        Z[:m,:m] = idwt(Z[:m,:m])
        m = m * 2

    return Z

# Equal MSE DWT. Returns compression ratio, Z
def gen_dwt_equal_mse(X, num_levels):
    energies = impulse_energies(num_levels)

    ssr_list = [1]
    for i in range(num_levels):
        ssr_list.append(np.sqrt(energies[0]/energies[i+1]))

    initial_ss = find_step_equal_rms_dwt(X, num_levels, ssr_list)

    ss_list = [initial_ss]
    for i in range(num_levels):
        ss_list.append(ss_list[0] * ssr_list[i+1])
    ss_arr = np.tile(np.array(ss_list), (3, 1))

    Y = nlevdwt(X, num_levels)
    Yq, Yq_ent_arr = quantdwt(Y, ss_arr)
    Yq_ent = np.sum(Yq_ent_arr)
    X_quant = quantise(X, 17)
    comp_ratio = entropy(X_quant) / Yq_ent
    Z = nlevidwt(Yq, num_levels)

    return comp_ratio, Z