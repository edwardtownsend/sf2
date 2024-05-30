import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import *
from cued_sf2_lab.dct import dct_ii
from front_end_schemes import *

def run_tests():
    test_results = []

    def assertEqual(a, b, test_name):
        if a != b:
            test_results.append(f"FAIL: {test_name} - {a} != {b}")
        else:
            test_results.append(f"PASS: {test_name} - {a} = {b}")

    X_pre_zero_mean, _ = load_mat_img(img='lighthouse.mat', img_info='X')
    X = X_pre_zero_mean - 128.0
    Xb_pre_zero_mean, _ = load_mat_img(img='bridge.mat', img_info='X')
    Xb = Xb_pre_zero_mean - 128.0
    Xf_pre_zero_mean, _ = load_mat_img(img='flamingo.mat', img_info='X')
    Xf = Xf_pre_zero_mean - 128.0

    # Test 8x8 DCT
    assertEqual(round(find_step_equal_rms_dct_lbt(X, dct_ii(8)), 1), 23.7, "Test find_step_equal_rms_dct_lbt() on 8x8 DCT")
    # Note the comp. ratio for DCT is greater than what you have in your report because you are using N=16 in dctbpp() instead of N=8 for consistency with what you did with LBT
    assertEqual(round(gen_dct_lbt_equal_rms(X, 8)[0], 3), 3.216, "Test 8x8 DCT compression ratio")
    Z = gen_dct_lbt_equal_rms(X, 8)[1]
    assertEqual(round(np.std(X - Z), 2), 4.86, "Test 8x8 DCT RMS error")
    
    # Test 4x4 LBT
    assertEqual(round(gen_dct_lbt_equal_rms(X, 4, s=np.sqrt(2))[0], 3), 3.564, "Test 4x4 LBT compression ratio")

    # Test 8x8 LBT
    assertEqual(round(gen_dct_lbt_equal_rms(X, 8, s=np.sqrt(2))[0], 3), 3.421, "Test 8x8 LBT compression ratio")

    # Test DWT
    assertEqual(round(gen_dwt_equal_mse(X, 3)[0], 3), 2.971, "Test DWT (equal MSE) compression ratio")

    for result in test_results:
        print(result)

run_tests()

"""
X_pre_zero_mean, _ = load_mat_img(img='lighthouse.mat', img_info='X')
X = X_pre_zero_mean - 128.0
quant_X = quant1(X, 20)
print(X[0:16, 0:16])
print(quant_X[0:16, 0:16])
print(X[:, :200].shape)
"""