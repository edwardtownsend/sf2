import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.dct import colxfm, dct_ii
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.laplacian_pyramid import bpp, quantise

def X_quant_entropy(X, step_size=17):
    X_quant = quantise(X, 17)
    return bpp(X_quant) * X_quant.size

def energy(X):
    return np.sum(X ** 2.0)

# LBT functions
def dctbpp(Yr, N):
    m, n = Yr.shape
    if m % N != 0 or n % N != 0:
        raise ValueError('Height/width of Yr not multiple of N')
    
    entropy_sum = 0
    for i in range(0, m, m//N):
        for j in range(0, n, n//N):
            sub_img = Yr[i:i+m//N, j:j+n//N]
            sub_img_ent = bpp(sub_img) * sub_img.size
            entropy_sum += sub_img_ent

    return entropy_sum

def gen_Y_quant_lbt(X, step_size, C, s):
    N = C.shape[0]
    Pf, Pr = pot_ii(N, s)
    t = np.s_[N//2:-N//2]

    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T

    Y = colxfm(colxfm(Xp, C).T, C).T
    Yq = quantise(Y, step_size)

    return Yq

def gen_Z_quant_lbt(X, step_size, C, s):
    N = C.shape[0]
    Pf, Pr = pot_ii(N, s)
    t = np.s_[N//2:-N//2] 

    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T

    Y = colxfm(colxfm(Xp, C).T, C).T
    Yq = quantise(Y, step_size)

    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    Zp = Z.copy()
    Zp[:, t] = colxfm(Zp[:, t].T, Pr.T).T
    Zp[t, :] = colxfm(Zp[t, :], Pr.T)

    return Zp

def compute_err_lbt(X, step_size, C, s):
    Zp = gen_Z_quant_lbt(X, step_size, C, s)
    return np.std(X - Zp)

def find_step_equal_rms(X, C, s):
    target_err = np.std(X - quantise(X, 17))

    # Binary search
    low, high = 15, 30
    while high - low > 0.1:
        mid = (low + high) / 2
        err = compute_err_lbt(X, mid, C, s)

        if err < target_err:
            low = mid
        else:
            high = mid

    return (low + high) / 2