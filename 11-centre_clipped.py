import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise
from front_end_schemes import *

X_non_zero_mean, _ = load_mat_img(img='lighthouse.mat', img_info='X')
Xb_non_zero_mean, _ = load_mat_img(img='bridge.mat', img_info='X')
Xf_non_zero_mean, _ = load_mat_img(img='flamingo.mat', img_info='X')

"""
COMMENTS:
Preferred front-end schemes are DCT (equal RMS, 8x8) LBT (equal RMS, 4x4 and 8x8) and DWT (equal MSE)
Need to fix DWT function as currently does not work


# GRAPH: Modifying rise1 and investigating effects on RMS error and comp. ratio
X = X_non_zero_mean - 128.0
rise1_ratios = list(np.arange(0.5, 2.0, 0.25))
block_sizes_lbt = [4, 8]
comp_ratios_dct, rms_errors_dct = [], []
comp_ratios_lbt, rms_errors_lbt = {4: [], 8: []}, {4: [], 8: []}
comp_ratios_dwt, rms_errors_dwt = [], []

for rise1 in rise1_ratios:
    # DCT
    comp_ratio, Z = gen_dct_lbt_equal_rms(X, 8, s=None, rise1_ratio = rise1)
    comp_ratios_dct.append(comp_ratio)
    rms_errors_dct.append(np.std(X-Z))
    
    # LBT
    for block_size in block_sizes_lbt:
        comp_ratio, Z = gen_dct_lbt_equal_rms(X, block_size, s=np.sqrt(2), rise1_ratio = rise1)
        comp_ratios_lbt[block_size].append(comp_ratio)
        rms_errors_lbt[block_size].append(np.std(X-Z))

    # DWT
    comp_ratio, Z = gen_dwt_equal_mse(X, 8)
    comp_ratios_dwt.append(comp_ratio)
    rms_errors_dwt.append(np.std(X-Z))

plt.figure(figsize=(8, 6))

# Plot DCT
plt.plot(rise1_ratios, comp_ratios_dct, label=f'Comp. Ratio 8x8 DCT')
plt.plot(rise1_ratios, rms_errors_dct, label=f'RMS Error 8x8 DCT')

# Plot LBT
for block_size in block_sizes_lbt:
    plt.plot(rise1_ratios, comp_ratios_lbt[block_size], label=f'Comp. Ratio {block_size}x{block_size} LBT')
    plt.plot(rise1_ratios, rms_errors_lbt[block_size], label=f'RMS Error {block_size}x{block_size} LBT')

# Plot DWT
plt.plot(rise1_ratios, comp_ratios_dwt, label=f'Comp. Ratio DWT (equal MSE)')
plt.plot(rise1_ratios, rms_errors_dwt, label=f'RMS Error DWT (equal MSE)')

plt.xlabel('Rise Ratios')
plt.ylabel('')
plt.title('Comp. Ratios and RMS Errors vs Rise Ratio for Selected Schemes')
plt.legend()
plt.grid(True)
plt.show()


# Modifying rise1 for LBT - image plots
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
plot_image(suppress_components(X, 8, 58))
plt.show()

supp_comp_nums = list(range(0, 63, 4))
comp_ratios = []
for i in supp_comp_nums:
    comp_ratios.append(gen_dct_lbt_equal_rms(X, 8, np.sqrt(2), 0.5, i)[0])

print(gen_dct_lbt_equal_rms(X, 8, np.sqrt(2), 0.5)[0])

plt.plot(supp_comp_nums, comp_ratios)
plt.xlabel('Number of components suppressed per image')
plt.ylabel('Compression Ratios')
plt.title('Graph 2')
plt.show()