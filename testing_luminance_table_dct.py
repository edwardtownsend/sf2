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


X_quant_ent = entropy(quantise(X, 17))
Y = forward_dct_lbt(X, C8)
Yq = quantise_jpeg(Y, step_table)
Z = inverse_dct_lbt(Yq, C8)
Yr = regroup(Yq, 8)
Yr_ent = dctbpp(Yr, 8)
comp_ratio = X_quant_ent / Yr_ent
rms_error = np.std(X-Z)
print(comp_ratio, rms_error)