import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.dct import colxfm, dct_ii, regroup
from cued_sf2_lab.lbt import pot_ii
from useful_functions import *

def gen_lbt(X, block_size, s):
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        return "Error the block size is not a power of two!"

    C = dct_ii(block_size)
    step_size = find_step_equal_rms_lbt(X, C, s)
    Yq = gen_Y_quant_lbt(X, step_size, C, s)
    Yr = regroup(Yq, 4)
    Yr_ent = dctbpp(Yr, 16)
    comp_ratio =  X_quant_entropy(X) / Yr_ent
    
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    Zp = Z.copy()
    Pf, Pr = pot_ii(block_size, s)
    t = np.s_[block_size//2:-block_size//2]
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)

    return comp_ratio, Zp