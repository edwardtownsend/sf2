import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.dct import colxfm, dct_ii
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from cued_sf2_lab.dwt import dwt, idwt

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

def find_step_equal_rms_lbt(X, C, s):
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

# DWT functions
def nlevdwt(X, n):
    Y = X.copy()
    m = X.shape[0]
    for i in range(n):
        Y[:m,:m] = dwt(Y[:m,:m])
        m //= 2
    
    return Y

def nlevidwt(Y, n):
    m = Y.shape[0] // (2 ** (n - 1))
    X = Y.copy()
    for i in range(n):
        X[:m,:m] = idwt(X[:m,:m])
        m = m * 2

    return X

def entropy(X):
    return bpp(X) * X.size

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray):
    Yq = Y.copy()
    dwtent = np.zeros(dwtstep.shape)
    
    num_levels = dwtstep.shape[1] - 1
    m = Y.shape[0]
    
    for i in range(num_levels):
        for j in range(3):
            if j == 0:
                sub_img = quantise(Yq[0:m//2, m//2:m], dwtstep[j, i])
                dwtent[j, i] = entropy(sub_img)
                Yq[0:m//2, m//2:m] = sub_img
            elif j == 1:
                sub_img = quantise(Yq[m//2:m, 0:m//2], dwtstep[j, i])
                dwtent[j, i] = entropy(sub_img)
                Yq[m//2:m, 0:m//2] = sub_img 
            elif j == 2:
                sub_img = quantise(Yq[m//2:m, m//2:m], dwtstep[j, i])
                dwtent[j, i] = entropy(sub_img)
                Yq[m//2:m, m//2:m] = sub_img
        m //= 2

    # Quantize the low-pass residual subband
    Yq[:m, :m] = quantise(Yq[:m, :m], dwtstep[0, num_levels])
    dwtent[0, num_levels] = entropy(Yq[:m, :m])

    return Yq, dwtent

def impulse_energies(n_levels):
    layer_energies = []

    for i in range(n_levels):
        Y = np.zeros((256, 256))
        # Top right image at level i
        centre_pixel_1 = [(256 // (2 ** i)) * 0.25, (256 // (2 ** i)) * 0.75]
        # Bottom right image at level i
        centre_pixel_2 = [(256 // (2 ** i)) * 0.75, (256 // (2 ** i)) * 0.75]
        # Bottom left image at level i
        centre_pixel_3 = [(256 // (2 ** i)) * 0.75, (256 // (2 ** i)) * 0.25]
        
        Y[int(centre_pixel_1[0]), int(centre_pixel_1[1])] = 100
        Y[int(centre_pixel_2[0]), int(centre_pixel_2[1])] = 100
        Y[int(centre_pixel_3[0]), int(centre_pixel_3[1])] = 100 

        Z0 = nlevidwt(Y, n_levels)  # Assuming nlevidwt is defined elsewhere
        layer_energies.append(energy(Z0))
    
    Y = np.zeros((256, 256))
    centre_pixel = [(256 // (2 ** n_levels)) * 0.25, (256 // (2 ** n_levels)) * 0.25]
    Y[int(centre_pixel[0]), int(centre_pixel[1])] = 100
    Z0 = nlevidwt(Y, n_levels)  # Assuming nlevidwt is defined elsewhere
    layer_energies.append(energy(Z0))

    return layer_energies