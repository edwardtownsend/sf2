import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.dct import colxfm, dct_ii
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from cued_sf2_lab.dwt import dwt, idwt

# General functions
def energy(X):
    return np.sum(X ** 2.0)

# Note: do not use entropy(X) for DCT/LBT
def entropy(X):
    return bpp(X) * X.size

# DCT/LBT functions
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

def gen_Y_quant_dct_lbt(X, step_size, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
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
    Yq = quantise(Y, step_size, rise1_ratio*step_size)

    return Yq

def gen_Z_quant_dct_lbt(X, step_size, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
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
    Yq = quantise(Y, step_size, rise1_ratio*step_size)
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    
    if s == None:
        return Z
    else:
        Zp = Z.copy()
        Zp[:, t] = colxfm(Zp[:, t].T, Pr.T).T
        Zp[t, :] = colxfm(Zp[t, :], Pr.T)
        return Zp

def compute_err_dct_lbt(X, step_size, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
    """
    Note C is the DCT matrix of coefficients, generated using dct_ii(N).
    C is input paramter instead of block_size to avoid having to re-compute C in every function when one function calls another.
    When s = None we perform the DCT instead of the LBT.
    """
    Zp = gen_Z_quant_dct_lbt(X, step_size, C, s, rise1_ratio, supp_comp_num)
    return np.std(X-Zp)

def find_step_equal_rms_dct_lbt(X, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
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
        err = compute_err_dct_lbt(X, mid, C, s, rise1_ratio, supp_comp_num)

        if err < target_err:
            low = mid
        else:
            high = mid

    return (low + high) / 2

def find_step_equal_rms_dct_lbt(X, C, s=None, rise1_ratio=0.5, supp_comp_num=0):
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
        err = compute_err_dct_lbt(X, mid, C, s, rise1_ratio, supp_comp_num)

        if err < target_err:
            low = mid
        else:
            high = mid

    return (low + high) / 2    

data = [
    [16, 11, 10, 16, 124, 140, 151, 161],
    [12, 12, 14, 19, 126, 158, 160, 155],
    [14, 13, 16, 24, 140, 157, 169, 156],
    [14, 17, 22, 29, 151, 187, 180, 162],
    [18, 22, 37, 56, 168, 109, 103, 177],
    [24, 35, 55, 64, 181, 104, 113, 192],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 199]
]


def suppress_components(Y, block_size, num_components):
    """
    Sets elements in Yq to zero in a zig-zag pattern based upon the JPEG documentation.

    Algorithm:
    The function iterates through the array Yq in a predefined pattern based on the direction:
    - "left": moves leftwards
    - "up": moves upwards
    - "up_right": moves diagonally up-right
    - "down_left": moves diagonally down-left

    Each element visited in Yq is set to zero, modifies in place.
    """
    if num_components == 0:
        return Y
    
    num_blocks = int(Y.shape[0] / block_size)
    start_row = block_size - 1
    start_col = block_size - 1
    curr_row = start_row
    curr_col = start_col
    direction = "left"

    for i in range(num_components):
        for j in range(num_blocks):
            for k in range(num_blocks):
                Y[curr_row + j*block_size, curr_col + k*block_size] = 0
                
        if direction == "left":
            curr_col -= 1

            if curr_row == start_row:
                direction = "up_right"
            if curr_row == 0:
                direction = "down_left"

        elif direction == "up_right":
            curr_row -= 1
            curr_col += 1

            if curr_row == 0:
                direction = "left"
            elif curr_col == start_col:
                direction = "up"


        elif direction == "up":
            curr_row -= 1
            
            if curr_col == start_col:
                direction = "down_left"
            if curr_col == 0:
                direction = "up_right"

        elif direction == "down_left":
            curr_row += 1
            curr_col -= 1

            if curr_row == start_row:
                direction = "left"
            if curr_col == 0:
                direction = "up"

    return Y


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

        Z0 = nlevidwt(Y, n_levels) 
        layer_energies.append(energy(Z0))
    
    Y = np.zeros((256, 256))
    centre_pixel = [(256 // (2 ** (n_levels - 1))) * 0.25, (256 // (2 ** (n_levels - 1))) * 0.25]
    Y[int(centre_pixel[0]), int(centre_pixel[1])] = 100
    Z0 = nlevidwt(Y, n_levels) 
    layer_energies.append(energy(Z0))

    return layer_energies

def compute_err_dwt(X, ss_arr, n_levels):
    Y = nlevdwt(X, n_levels)
    Yq = quantdwt(Y, ss_arr)[0]
    Z = nlevidwt(Yq, n_levels)
    
    return np.std(X-Z)

def find_step_equal_rms_dwt(X, n_levels, ssr_list):
    target_err = np.std(X - quantise(X, 17))

    # Binary search
    low, high = 5, 30
    while high - low > 0.1:
        mid = (low + high) / 2
        ss_list = [mid]
        for i in range(n_levels):
            ss_list.append(mid * ssr_list[i+1])
        ss_arr = np.tile(np.array(ss_list), (3, 1))
        err = compute_err_dwt(X, ss_arr, n_levels)

        if err < target_err:
            low = mid
        else:
            high = mid

    return (low + high) / 2