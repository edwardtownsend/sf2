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

def find_min_ssr_jpeg(X, step_table_type, C, s=None):
    # Binary search
    low, high = 0, 200
    while high - low > 0.1:
        mid = (low + high) / 2

        vlctemp, _ = jpegenc_dct_lbt(X, mid, step_table_type, C, s)
        num_bits = vlctemp[:,1].sum()

        if num_bits < 40000:
            high = mid
        else:
            low = mid

    return (low + high) / 2    

def compute_scores_dct_lbt(X, ssr, step_table_type, C, s=None):
    vlc, _ = jpegenc_dct_lbt(X, ssr, step_table_type, C, s)
    num_bits = vlc[:,1].sum()
    Z = jpegdec_dct_lbt(vlc, ssr, step_table_type, C, s)
    
    rms_err = np.std(Z - X)
    ssim_score = ssim(X, Z, data_range=255)
    if ssim_score == None:
        ssim_score = 0
    
    return rms_err, ssim_score, num_bits

def compress(X):
    C8 = dct_ii(8)
    # Compute min ssr to achieve 5kB size
    ssr_dct = find_min_ssr_jpeg(X, 1, C8, None)
    ssr_lbt = find_min_ssr_jpeg(X, 1, C8, np.sqrt(2))

    # Compute scores at these step sizes
    rms_dct, ssim_dct, bits_dct = compute_scores_dct_lbt(X, ssr_dct, 1, C8, None)
    rms_lbt, ssim_lbt, bits_lbt = compute_scores_dct_lbt(X, ssr_dct, 1, C8, np.sqrt(2))

    # Compute final decoded images at these step sizes
    vlc_dct, _ = jpegenc_dct_lbt(X, ssr_dct, 1, C8, None)
    Z_dct = jpegdec_dct_lbt(vlc_dct, ssr_dct, 1, C8, None)
    vlc_lbt, _ = jpegenc_dct_lbt(X, ssr_lbt, 1, C8, np.sqrt(2))
    Z_lbt = jpegdec_dct_lbt(vlc_lbt, ssr_lbt, 1, C8, np.sqrt(2))

    if ssim_dct > ssim_lbt:
        print(f"decoded with an rms error {rms_dct} using {bits_dct} bits and an SSIM score of {ssim_dct} (using DCT)")
        fig, ax = plt.subplots()
        plot_image(Z_dct, ax=ax)

    else:
        print(f"decoded with an rms error {rms_lbt} using {bits_lbt} bits and an SSIM score of {ssim_lbt} (using LBT)")
        fig, ax = plt.subplots()
        plot_image(Z_lbt, ax=ax)

X_pre_zero_mean, _ = load_mat_img(img='lighthouse.mat', img_info='X')
X = X_pre_zero_mean - 128.0
compress(X)