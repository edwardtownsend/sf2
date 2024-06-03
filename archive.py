"""
TRYING TO VARY THE STEP TABLE TO SEE IF CAN GET PER IMAGE BENEFITS

NEGILIGBLE BENEFITS - DON'T BOTHER ITERATING THIS FURTHER

opt_step_table = gen_step_table(0)
step_table = opt_step_table
C8 = dct_ii(8)
ssr_dct = find_min_ssr_jpeg(Xb, step_table, C8, None)
# rms_dct, ssim_dct = compute_scores_dct_lbt(Xb, ssr_dct * step_table, C8, None)[0:2]
print(ssr_dct)

opt_ssim_dct = ssim_dct
step_table_1 = gen_step_table(0)
step_table_2 = gen_step_table(2)
ssr_1 = find_min_ssr_jpeg(Xb, step_table_1, C8, None)
ssr_2 = find_min_ssr_jpeg(Xb, step_table_2, C8, None)
vlc_1 = jpegenc_dct_lbt(Xb, ssr_1*step_table_1, C8, s=None, N=8, M=8, opthuff=False, dcbits=8)[0]
vlc_2 = jpegenc_dct_lbt(Xb, ssr_2*step_table_2, C8, s=None, N=8, M=8, opthuff=False, dcbits=8)[0]
Z_1 = jpegdec_dct_lbt(vlc_1, ssr_1*step_table_1, C8)
Z_2 = jpegdec_dct_lbt(vlc_2, ssr_2*step_table_2, C8)

fig, axes = plt.subplots(1, 2)
plot_image(Z_1, ax=axes[0])
plot_image(Z_2, ax=axes[1])
plt.show()

img = Xf
opt_step_table = gen_step_table(0)
C8 = dct_ii(8)
ssr_dct = find_min_ssr_jpeg(img, opt_step_table, C8, None)
opt_ssim_dct = compute_scores_dct_lbt(img, ssr_dct * opt_step_table, C8, None)[1]

start_time = time.time()
for i in range(1):
    for j in range(4):
        ssr_dct = find_min_ssr_jpeg(img, opt_step_table, C8, None)
        curr_opt_k = 0
        curr_ssim_diff = 0

        curr_step = int(opt_step_table[i, j])
        if curr_step > 15:
            curr_step = 15
        for k in range((3 - curr_step), 5):
            print(i, j, k)
            step_table = opt_step_table.copy()
            step_table[i, j] += k
            ssim_dct = compute_scores_dct_lbt(img, ssr_dct * step_table, C8, None)[1]
            if ssim_dct - opt_ssim_dct > curr_ssim_diff:
                print(ssim_dct)
                curr_opt_k = k
                curr_ssim_diff = ssim_dct - opt_ssim_dct
        
        opt_step_table[i, j] += curr_opt_k
        opt_ssim_dct += curr_ssim_diff

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
print(opt_ssim_dct)
print(opt_step_table)
print(compute_scores_dct_lbt(img, ssr_dct * opt_step_table, C8, None)[0:2])



ATTEMPTING TO IMPLEMENT DWT ENCODING

X_22 = np.round(X_22).astype(np.int32)  
Yq_dwt = gen_Y_dwt_equal_mse(X_22, 3)[0]
Yq = dwtgroup(Yq_dwt, 3)
print(Yq[:16, :8])

pred_num_bits = gen_Y_dwt_equal_mse(X_22, 3)[1]
print(f"Predicted number of bits to encode using dwt is {pred_num_bits}")
vlc, _ = jpegenc_dwt(X_22, 3, N=8, M=8, opthuff=True, dcbits=8)
actual_num_bits = vlc[:,1].sum()
print(f"Actual number of bits to encode using dwt is {actual_num_bits}")



INVESTIGATING VARYING LUMINANCE TABLE WITH ENERGIES OF FREQUENCIES PRESENT IN IMAGE
K IS SET TO MAXIMUM VALUE WITHOUT BREAKING ENCODER FUNCTION

# bot_frequencies graph for X_23 using k = 0.0000001
bot_frequencies = list(range(1, 9))
ssim_scores = []
for i in bot_frequencies:
    ssim_scores.append(compress(X_20, 0, i, 0.0000001))
plt.plot(bot_frequencies, ssim_scores)
plt.show()



USING ENERGY MATRIX TO ALTER AC LUMINANCE TABLE IN COMPRESS - NEGLIGIBLE INCREASE IN SSIM

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




VARYING S FOR LBT    

rms_lbt_list = []
ssim_lbt_list = []
s = list(np.arange(1, 2, 0.1))
for i in s:
    print(i)
    rms_lbt, ssim_lbt = compress(X_23, 0, i, 4, 16)
    rms_lbt_list.append(rms_lbt)
    ssim_lbt_list.append(ssim_lbt)

plt.plot(s, rms_lbt_list, label="rms")
plt.plot(s, ssim_lbt_list, label="ssim")
plt.legend()
plt.show()    


TRYING TO ADD UP THE QUANTISATION ERRORS AND THEN ADJUST THE QUANTISATION TABLE ACCORDINGLY - NEGLIGIBLE IMPROVEMENTS

def compute_error_rate(img, step_table_type, s=None, N=8, M=8):
    errors = np.zeros((N, N))
    step_table = gen_step_table(step_table_type)
    C = dct_ii(N)

    Y = forward_dct_lbt(img, C, s)

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            errors[i % N, j % N] += abs((Y[i, j] % step_table[i % N, j % N]) - step_table[i % N, j % N])

    return errors

def compute_avg_error_rate(images, step_table_type, s=None, N=8, M=8):
    errors = np.zeros((N, N))
    for img in images:
        step_table = gen_step_table(step_table_type)
        C = dct_ii(N)

        Y = forward_dct_lbt(img, C, s)

        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                errors[i % N, j % N] += abs((Y[i, j] % step_table[i % N, j % N]) - step_table[i % N, j % N])

    return errors / len(images)

def adjust_step_table(img, step_table_type, avg_errors, s=None, N=8, M=8):
    errors = compute_error_rate(img, step_table_type, s, N, M)
    step_table = gen_step_table(step_table_type)

    for i in range(N):
        for j in range(N):
            if i != 0 or j != 0:
                step_factor = 1.0 + abs(errors[i, j] - avg_errors[i, j])/avg_errors[i, j]
                step_table[i, j] = round(step_table[i, j] / (step_factor * 1.5))

    return step_table

images = [Xb, Xf, X_23, X_22, X_21, X_20, X_19]
avg_errors = compute_avg_error_rate(images, 3, 1.0)
print(avg_errors)
new_step_table = adjust_step_table(X_20, 3, avg_errors, 1.0)
print(new_step_table)
print(compress(X_20, gen_step_table(3), 1.0))
print(compress(X_20, new_step_table, 1.0))

"""