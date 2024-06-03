import warnings
import inspect
import time
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

def find_min_ssr_jpeg(X, step_table, C, s=None, N=8, M=8, supp_comp_num=0):
    # Binary search
    def binary_search(ssr_low, ssr_high):
        while ssr_high - ssr_low > 0.001:
            ssr_mid = (ssr_low + ssr_high) / 2

            vlctemp, _ = jpegenc_dct_lbt(X, ssr_mid * step_table, C, s, N, M, False, 8, supp_comp_num)
            num_bits = vlctemp[:,1].sum()

            if num_bits < 40960:
                next_ssr_high = ssr_mid
                next_high_vlctemp, _ = jpegenc_dct_lbt(X, next_ssr_high * step_table, C, s, N, M, False, 8, supp_comp_num)
                next_high_num_bits = next_high_vlctemp[:,1].sum() 
                if next_high_num_bits > 40960:
                    return ssr_mid
                else:
                    ssr_high = ssr_mid
            else:
                ssr_low = ssr_mid
        
        if num_bits > 40960:
            return binary_search(ssr_low, ssr_high * 2.0)
        else:
            return ssr_high
    
    mean_step = np.mean(step_table)
    ssr_low, ssr_high = 0, 300 / mean_step
    return binary_search(ssr_low, ssr_high)

def compute_scores_dct_lbt(X, step_table, C, s=None, N=8, M=8, supp_comp_num=0):
    vlc, _ = jpegenc_dct_lbt(X, step_table, C, s, N, M, False, 8, supp_comp_num)
    num_bits = vlc[:,1].sum()
    Z = jpegdec_dct_lbt(vlc, step_table, C, s)
    
    rms_err = np.std(Z - X)
    ssim_score = ssim(X, Z, data_range=255)
    if ssim_score == None:
        ssim_score = 0
    
    return rms_err, ssim_score, num_bits

def compress(X, step_table_type, s=None, N=8, M=8, supp_comp_num=0):
    step_table = gen_step_table(step_table_type)
    C = dct_ii(N)

    # Compute min ssr to achieve 5kB size
    ssr = find_min_ssr_jpeg(X, step_table, C, s, N, M, supp_comp_num)

    # Compute scores at these step sizes
    rms, ssim, bits_dct = compute_scores_dct_lbt(X, ssr * step_table, C, s, N, M, supp_comp_num)
    
    return rms, ssim

"""
img = Xb
C8 = dct_ii(8)
opt_step_table = gen_step_table(0)
opt_ssr = find_min_ssr_jpeg(img, opt_step_table, C8, 1.0)
max_ssim = compute_scores_dct_lbt(img, opt_ssr * opt_step_table, C8, 1.0)[1]
print(max_ssim)

for i in range(1):
    for j in range(3):
        curr_step = int(opt_step_table[i, j])
        opt_k = 0

        for k in range(-curr_step + 5, -curr_step + 7):
            print(i, j, k)
            curr_step_table = opt_step_table.copy()
            curr_step_table[i, j] += k
            curr_ssr = find_min_ssr_jpeg(img, curr_step_table, C8, 1.0)
            curr_ssim = compute_scores_dct_lbt(img, curr_ssr * curr_step_table, C8, 1.0)[1]

            if curr_ssim > max_ssim:
                print(curr_ssim)
                opt_k = k
                max_ssim = curr_ssim
        
        opt_step_table[i, j] += opt_k

print(opt_step_table)
print(max_ssim)
print(compute_scores_dct_lbt(Xb, opt_step_table, C8, 1.0))
"""

print(compress(Xf, 3, 1.0, 8, 8, 44))

"""
print(compress(Xf, 3, 1.0))
print(compress(X_23, 3, 1.0))
print(compress(X_22, 3, 1.0))
print(compress(X_21, 3, 1.0))
print(compress(X_20, 3, 1.0))
print(compress(X_19, 3, 1.0))"""