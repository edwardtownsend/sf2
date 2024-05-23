import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise

lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

x = np.arange(-100, 100+1, 2)
print(x)
y = quantise(x, 20, 40)

"""
Preferred front-end schemes are LBT (equal RMS) and DWT (equal MSE)
Leaving equal MSE for now as not sure if have calculated correctly

Investigate 8x8 block size for LBT first, then potentially look at 4x4

Think about numerical parameters that can use to esimate the visual quality of an image without having to view it
"""
