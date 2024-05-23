import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise
from front_end_schemes import *

lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

"""
Preferred front-end schemes are LBT (equal RMS) and DWT (equal MSE)
Need to fix DWT function as currently does not work

Investigate 8x8 block size for LBT first, then potentially look at 4x4

Think about numerical parameters that can use to esimate the visual quality of an image without having to view it
"""

# Test front-end functions
comp_ratio_lbt, Zp = gen_lbt_equal_rms(lighthouse, 8, np.sqrt(2))

comp_ratio_dwt, Z = gen_dwt_equal_mse(lighthouse, 3)

ig, ax = plt.subplots()
plot_image(Zp, ax = ax)
plt.show()

fig, ax = plt.subplots()
plot_image(Z, ax = ax)
plt.show()