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
    [16, 11, 10, 16, 124, 140, 151, 161],
    [12, 12, 14, 19, 126, 158, 160, 155],
    [14, 13, 16, 24, 140, 157, 169, 156],
    [14, 17, 22, 29, 151, 187, 180, 162],
    [18, 22, 37, 56, 168, 109, 103, 177],
    [24, 35, 55, 64, 181, 104, 113, 192],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 199]
], dtype=np.float64)
"""
ones_array = np.ones((8, 8), dtype=np.float64)
step_table = ones_array*23

C8 = dct_ii(8)
qY = gen_Y_quant_dct_jpeg(X, step_table, C8, supp_comp_num=0)
Z = colxfm(colxfm(qY.T, C8.T).T, C8.T)
X_quant_ent = entropy(quantise(X, 17))
qY_ent = entropy(qY)
comp_ratio = X_quant_ent / qY_ent
print(comp_ratio)

fig, ax = plt.subplots()
plot_image(Z, ax=ax)
plt.show()

step_ratio = find_step_ratio_equal_rms_dct_jpeg(X, step_table, C8)
print(step_ratio)
print(np.std(quantise(X, 17) - X))
print(compute_err_dct_jpeg(X, step_ratio, step_table, C8))