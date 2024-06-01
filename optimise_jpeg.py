import warnings
import inspect
import numpy as np
import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import *
from cued_sf2_lab.dct import *
from cued_sf2_lab.bitword import bitword
from cued_sf2_lab.jpeg import *
from front_end_schemes import *
from useful_functions import *
from encoder_decoder import *

from skimage.metrics import structural_similarity as ssim

def find_min_ssr_jpeg(X, step_table, C, s=None):
    # Binary search
    def binary_search(ssr_low, ssr_high):
        while ssr_high - ssr_low > 0.0001:
            ssr_mid = (ssr_low + ssr_high) / 2

            vlctemp, _ = jpegenc_dct_lbt(X, ssr_mid, step_table, C, s)
            num_bits = vlctemp[:,1].sum()
            print(num_bits)

            if num_bits < 40960:
                next_ssr_high = ssr_mid
                next_high_vlctemp, _ = jpegenc_dct_lbt(X, next_ssr_high, step_table, C, s)
                next_high_num_bits = next_high_vlctemp[:,1].sum() 
                if next_high_num_bits > 40960:
                    return ssr_mid
                else:
                    ssr_high = ssr_mid
            else:
                ssr_low = ssr_mid
        
        if num_bits > 40960:
            return binary_search(ssr_low, ssr_high*2.0)
        else:
            return ssr_high
    
    mean_step = np.mean(step_table)
    ssr_low, ssr_high = 0, 200 / mean_step
    return binary_search(ssr_low, ssr_high)

def compute_scores_dct_lbt(X, ssr, step_table, C, s=None):
    vlc, _ = jpegenc_dct_lbt(X, ssr, step_table, C, s)
    num_bits = vlc[:,1].sum()
    Z = jpegdec_dct_lbt(vlc, ssr, step_table, C, s)
    
    rms_err = np.std(Z - X)
    ssim_score = ssim(X, Z, data_range=255)
    if ssim_score == None:
        ssim_score = 0
    
    return rms_err, ssim_score, num_bits

def compress_1(X, step_table_type):
    step_table = gen_step_table(step_table_type)
    C8 = dct_ii(8)
    # Compute min ssr to achieve 5kB size
    ssr_dct = find_min_ssr_jpeg(X, step_table, C8, None)
    ssr_lbt = find_min_ssr_jpeg(X, step_table, C8, np.sqrt(2))

    # Compute scores at these step sizes
    rms_dct, ssim_dct, bits_dct = compute_scores_dct_lbt(X, ssr_dct, step_table, C8, None)
    rms_lbt, ssim_lbt, bits_lbt = compute_scores_dct_lbt(X, ssr_lbt, step_table, C8, np.sqrt(2))

    # Compute final decoded images at these step sizes
    vlc_dct, _ = jpegenc_dct_lbt(X, ssr_dct, step_table, C8, None)
    Z_dct = jpegdec_dct_lbt(vlc_dct, ssr_dct, step_table, C8, None)
    vlc_lbt, _ = jpegenc_dct_lbt(X, ssr_lbt, step_table, C8, np.sqrt(2))
    Z_lbt = jpegdec_dct_lbt(vlc_lbt, ssr_lbt, step_table, C8, np.sqrt(2))

    if ssim_dct < ssim_lbt:
        print(f"decoded with an rms error {rms_dct} using {bits_dct} bits and an SSIM score of {ssim_dct} (using DCT)")
        fig, ax = plt.subplots()
        plot_image(Z_dct, ax=ax)
        plt.show()

    else:
        print(f"decoded with an rms error {rms_lbt} using {bits_lbt} bits and an SSIM score of {ssim_lbt} (using LBT)")
        fig, ax = plt.subplots()
        plot_image(Z_lbt, ax=ax)
        plt.show()

def compress_2(X, step_table_type, bot_freq, k=0):
    step_table = gen_step_table(step_table_type)
    C = dct_ii(8)

    # Compute energy matrix
    N = C.shape[0]
    energy_arr = np.zeros((N, N))
    Y = forward_dct_lbt(X, C)
    Yr = regroup(Y, N)
    Yr_block_size = Y.shape[0] // N

    for i in range(0, N):
        for j in range(0, N):
            sub_img = Yr[i * Yr_block_size : (i+1) * Yr_block_size, j * Yr_block_size: (j+1) * Yr_block_size]
            energy_sub_img = np.sum(sub_img ** 2.0)
            energy_arr[i, j] = energy_sub_img

    # Normalise energy_matrix to have mean of 1. Set k to vary the spread of energies. Smaller k, lower spread.
    zero_mean_arr = energy_arr - np.mean(energy_arr)
    zero_mean_arr[:bot_freq, :bot_freq] = 0
    arr_shruken = zero_mean_arr * k
    norm_energy_arr = arr_shruken + 1.0
    norm_energy_arr[:bot_freq, :bot_freq] = 1.0

    step_table /= norm_energy_arr
    print(step_table)

    # Compute min ssr to achieve 5kB size
    ssr_dct = find_min_ssr_jpeg(X, step_table, C, None)

    # Compute scores at these step sizes
    rms_dct, ssim_dct, bits_dct = compute_scores_dct_lbt(X, ssr_dct, step_table, C, None)
    
    print(rms_dct, ssim_dct, bits_dct)

X_pre_zero_mean, _ = load_mat_img(img='images/lighthouse.mat', img_info='X')
X = X_pre_zero_mean - 128.0
Xb_pre_zero_mean, _ = load_mat_img(img='images/bridge.mat', img_info='X')
Xb = Xb_pre_zero_mean - 128.0
Xf_pre_zero_mean, _ = load_mat_img(img='images/flamingo.mat', img_info='X')
Xf = Xf_pre_zero_mean - 128.0
X_19_pre_zero_mean, _ = load_mat_img(img='images/2019.mat', img_info='X')
X_19 = X_19_pre_zero_mean - 128.0
X_20_pre_zero_mean, _ = load_mat_img(img='images/2020.mat', img_info='X')
X_20 = X_20_pre_zero_mean - 128.0
X_21_pre_zero_mean, _ = load_mat_img(img='images/2021.mat', img_info='X')
X_21 = X_21_pre_zero_mean - 128.0
X_22_pre_zero_mean, _ = load_mat_img(img='images/2022.mat', img_info='X')
X_22 = X_22_pre_zero_mean - 128.0
X_23_pre_zero_mean, _ = load_mat_img(img='images/2023.mat', img_info='X')
X_23 = X_23_pre_zero_mean - 128.0

# compress_1(Xb, 0)
compress_1(X_22, 0)