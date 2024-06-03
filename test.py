import warnings
import inspect
import time
import numpy as np
import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import *
from cued_sf2_lab.dct import *
from cued_sf2_lab.bitword import bitword
from cued_sf2_lab.jpeg import *
from front_end_schemes import *
from useful_functions import *
from encoder_decoder import *

"""
C = dct_ii(8)

Xb_pre_zero_mean, _ = load_mat_img(img='images/lighthouse.mat', img_info='X')
Xb = Xb_pre_zero_mean - 128.0

# Compute energy matrix
N = C.shape[0]
energy_arr = np.zeros((N, N))
Y = forward_dct_lbt(Xb, C)
Yr = regroup(Y, N)
Yr_block_size = Y.shape[0] // N

for i in range(0, N):
    for j in range(0, N):
        sub_img = Yr[i * Yr_block_size : (i+1) * Yr_block_size, j * Yr_block_size: (j+1) * Yr_block_size]
        energy_sub_img = np.sum(sub_img ** 2.0)
        energy_arr[i, j] = energy_sub_img

print(energy_arr)
"""

