import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from front_end_schemes import *
from cued_sf2_lab.familiarisation import load_mat_img, plot_image

X_non_zero_mean, _ = load_mat_img(img='lighthouse.mat', img_info='X')
X = X_non_zero_mean - 128.0
Zp = gen_dct_lbt_equal_rms(X, 8, np.sqrt(2))[1]

fig, axes = plt.subplots(1, 2)
plot_image(X, ax=axes[0])
plot_image(Zp, ax=axes[1])
plt.show()