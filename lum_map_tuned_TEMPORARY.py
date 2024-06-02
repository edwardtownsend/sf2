## Add this into the encoder decoder function when ready
## Will tune image down to below 5kB and has option to do LBT pre and post filtering
## Returns decoded image, number of bits and SSIM score

def tuneDCTmap(X, s=None):
    C8 = dct_ii(8)
    X = X - 128.0
    if s == None:
        Xp = X
    else:
        N = C8.shape[0]
        Pf, Pr = pot_ii(N, s)
        t = np.s_[N//2:-N//2]
        Xp = X.copy()
        Xp[t, :] = colxfm(Xp[t, :], Pf)
        Xp[:, t] = colxfm(Xp[:, t].T, Pf).T    
    calc_bits = 100000
    k = 2
    while calc_bits > 40960:
        k += 0.1
        vlc, _ = jpegenc_dct_lbt(Xp, k*step, C8)
        calc_bits = vlc[:,1].sum()
    Z = jpegdec_dct_lbt(vlc, k*step, C8)
    if s == None:
        Zp = Z
    else:
        N = C8.shape[0]
        Pf, Pr = pot_ii(N, s)
        t = np.s_[N//2:-N//2]
        Zp = Z.copy()
        Zp[:, t] = colxfm(Zp[:, t].T, Pr.T).T
        Zp[t, :] = colxfm(Zp[t, :], Pr.T)
    score = visualerror(X, Zp)
    rms_err = np.std(Z-X)

    return Zp, calc_bits, score, rms_err


### Final image compression function? If DWT isnt working we can simplify it down to choose between LBT and DCT
def image_compression(X):
    Z1, B1, S1, e1 = tuneDCTmap(X, s=None)
    Z2, B2, S2, e1 = tuneDCTmap(X, s=np.sqrt(2))  #We can choose an s value on the morning of?
    """ potential line of code if DWT is working
    Z3, B3, S3, e3 = LBTFUNCTION() """
    if S1 > S2:
        if S1 > S3:
            print(f"Image using DCT method, {B1} bits, rms error of {e1} and SSIM score of {S1}")
            fig, ax = plt.subplots()
            plot_image(Z1, ax=ax)
        elif S3 > S1:
            print(f"Image using DCT method, {B3} bits, rms error of {e3} and SSIM score of {S3}")
            fig, ax = plt.subplots()
            plot_image(Z3, ax=ax)
    elif S2 > S1:
        if S2 > S3:
            print(f"Image using DCT method, {B2} bits, rms error of {e2} and SSIM score of {S2}")
            fig, ax = plt.subplots()
            plot_image(Z2, ax=ax)        
        elif S3 > S2:
            print(f"Image using DCT method, {B3} bits, rms error of {e3} and SSIM score of {S3}")
            fig, ax = plt.subplots()
            plot_image(Z3, ax=ax)
