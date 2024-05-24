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


"""
# Modifying rise1 for LBT
X = lighthouse - 128.0
rise1_ratios = list(np.arange(0.1, 3, 0.05))
comp_ratios = []
for i in rise1_ratios:
    comp_ratios.append(gen_lbt_equal_rms(X, 16, np.sqrt(2), i)[0])

plt.plot(rise1_ratios, comp_ratios)
plt.xlabel('Rise Ratios')
plt.ylabel('Compression Ratios')
plt.title('Compression Ratio vs Rise Ratio for LBT (equal RMS)')
plt.show()
"""
"""
Can investigate further for LBT:
    1 Effect of block size on this graph
    2 Effect of scaling factor on this graph (think this will be limited)
"""

# Suppressing highest horizontal/vertical frequency sub-images - could look in zig-zag manner as seen in JPEG standard in notebook 12


fig, ax = plt.subplots()
plot_image(suppress_components(lighthouse, 8, 58))
plt.show()

X = lighthouse - 128.0
supp_comp_nums = list(range(0, 63, 4))
comp_ratios = []
for i in supp_comp_nums:
    comp_ratios.append(gen_lbt_equal_rms(X, 8, np.sqrt(2), 0.5, i)[0])

print(gen_lbt_equal_rms(X, 8, np.sqrt(2), 0.5)[0])

plt.plot(supp_comp_nums, comp_ratios)
plt.xlabel('Number of components suppressed per image')
plt.ylabel('Compression Ratios')
plt.title('Graph 2')
plt.show()

fig, ax = plt.subplots()
plot_image(gen_lbt_equal_rms(X, 8, np.sqrt(2), 0.5, 58)[1])
plt.show()