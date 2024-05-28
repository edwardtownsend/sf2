import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.jpeg import *
from front_end_schemes import *

arr = np.random.randint(0, 3, 20)
rumampl_encoded_arr = runampl(arr)

scan = diagscan(8)
print(scan)