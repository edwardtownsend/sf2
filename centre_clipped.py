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
COMMENTS

Preferred front-end schemes are LBT (equal RMS) and DWT (equal MSE)
Need to fix DWT function as currently does not work

# Modifying rise1 for LBT - graph
X = lighthouse - 128.0
rise1_ratios = list(np.arange(0.5, 2.0, 0.25))
block_sizes = [4, 8, 16]
comp_ratios = {4: [], 8: [], 16: []}
rms_errors = {4: [], 8: [], 16: []}

for rise1 in rise1_ratios:
    for block_size in block_sizes:
        comp_ratio, Z = gen_dct_lbt_equal_rms(X, block_size, s=None, rise1_ratio = rise1)
        comp_ratios[block_size].append(comp_ratio)
        rms_errors[block_size].append(np.std(X-Z))

plt.figure(figsize=(8, 6))
for block_size in block_sizes:
    plt.plot(rise1_ratios, comp_ratios[block_size], label=f'Comp. Ratio C{block_size}')
    plt.plot(rise1_ratios, rms_errors[block_size], label=f'RMS Error C{block_size}')

plt.xlabel('Rise Ratios')
plt.ylabel('')
plt.title('Comp. Ratios and RMS Errors vs Rise Ratio for DCT (equal RMS)')
plt.legend(title='Block Size')
plt.grid(True)
plt.show()

Zp = gen_dct_lbt_equal_rms(X, 8, np.sqrt(2), 1.5)[1]
print(np.std(X-Zp))

"""

# Modifying rise1 for LBT - image plots
X = lighthouse - 128.0
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
plot_image(gen_dct_lbt_equal_rms(X, 4, np.sqrt(2), 0.5)[1], ax=ax[0])
ax[0].set_title('rise1 ratio = 0.5')
plot_image(gen_dct_lbt_equal_rms(X, 4, np.sqrt(2), 1.0)[1], ax=ax[1])
ax[1].set_title('rise1 ratio = 1.0')
plot_image(gen_dct_lbt_equal_rms(X, 4, np.sqrt(2), 1.5)[1], ax=ax[2])
ax[2].set_title('rise1 ratio = 1.5')
plt.tight_layout()
fig.suptitle('LBT for different rise1 ratios, using C4 and s = root(2)')
plt.show()

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
"""