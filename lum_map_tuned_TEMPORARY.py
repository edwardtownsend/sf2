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

    return Zp, calc_bits, score