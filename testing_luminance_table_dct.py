import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import *
from cued_sf2_lab.dct import dct_ii
from front_end_schemes import *
from luminance_table_dct import *
from useful_functions import *
from cued_sf2_lab.dct import dct_ii

X_pre_zero_mean, _ = load_mat_img(img='lighthouse.mat', img_info='X')
X = X_pre_zero_mean - 128.0

# JPEG quantisation luminance table (section K.1)

"""
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
"""
ones_array = np.ones((8, 8), dtype=np.float64)
step_table = ones_array*23

C = dct_ii(8)
step_ratio = find_step_ratio_equal_rms_dct_jpeg(X, step_table, C)
print(f"Step ratio is {step_ratio}")
print(f"rms error for this step ratio is {compute_err_dct_jpeg(X, step_ratio, step_table, C)}")
Z = gen_Z_quant_dct_jpeg(X, step_table*step_ratio, C)
print(f"Verify same rms error: {np.std(X-Z)}")

X_quant_ent = entropy(quantise(X, 17))
Yq = gen_Y_quant_dct_jpeg(X, step_table, C)
Yr = regroup(Yq, 8)
Yr_ent = dctbpp(Yr, 8)
comp_ratio = X_quant_ent / Yr_ent
print(f"Comp. ratio: {comp_ratio}")