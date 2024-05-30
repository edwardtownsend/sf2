from typing import Tuple, NamedTuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from .laplacian_pyramid import quant1, quant2
from .dct import dct_ii, colxfm, regroup
from .bitword import bitword
from .lbt import pot_ii
from .familiarisation import plot_image
from visstructure import visualerror
from useful_functions import *

def jpegenc_lbt(X, qstep, s=None, N=8, M=8, opthuff=False, dcbits=8):
    if M % N != 0:
        raise ValueError('M must be an integer multiple of N!')
    
    C8 = dct_ii(8)
    Y = gen_Y_dct_lbt(X, C8, s)
    Yq = quant1(Y, qstep, qstep).astype('int')

    scan = diagscan(M)
    dhufftab = huffdflt(1)
    huffcode, ehuf = huffgen(dhufftab)

    sy = Yq.shape
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r+M,c:c+M]
            if M > N:
                yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            dccoef = yqflat[0] + 2 ** (dcbits-1)
            if dccoef not in range(2**dcbits):
                raise ValueError('DC coefficients too large for desired number of bits')
            vlc.append(np.array([[dccoef, dcbits]]))
            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    if not opthuff:
        return vlc, dhufftab

    dhufftab = huffdes(huffhist)
    huffcode, ehuf = huffgen(dhufftab)

    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r+M, c:c+M]
            if M > N:
                yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            dccoef = yqflat[0] + 2 ** (dcbits-1)
            vlc.append(np.array([[dccoef, dcbits]]))
            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    return vlc, dhufftab

def jpegdec_lbt(vlc, qstep, s=None, N=8, M=8, hufftab=None, dcbits=8, W=256, H=256):
    if M % N != 0:
        raise ValueError('M must be an integer multiple of N!')

    scan = diagscan(M)
    opthuff = (hufftab is not None)
    if opthuff:
        if len(hufftab.bits.shape) != 1:
            raise ValueError('bits.shape must be (len(bits),)')
    else:
        hufftab = huffdflt(1)

    huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
    huffcode, ehuf = huffgen(hufftab)
    k = 2 ** np.arange(17)
    eob = ehuf[0]
    run16 = ehuf[15 * 16]
    i = 0
    Zq = np.zeros((H, W))

    for r in range(0, H, M):
        for c in range(0, W, M):
            yq = np.zeros(M**2)
            cf = 0
            if vlc[i, 1] != dcbits:
                raise ValueError('The bits for the DC coefficient does not agree with vlc table')
            yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)
            i += 1

            while np.any(vlc[i] != eob):
                run = 0
                while np.all(vlc[i] == run16):
                    run += 16
                    i += 1
                start = huffstart[vlc[i, 1] - 1]
                res = hufftab.huffval[start + vlc[i, 0] - huffcode[start]]
                run += res // 16
                cf += run + 1
                si = res % 16
                i += 1
                if vlc[i, 1] != si:
                    raise ValueError('Problem with decoding .. you might be using the wrong hufftab table')
                ampl = vlc[i, 0]
                thr = k[si - 1]
                yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)
                i += 1

            i += 1
            yq = yq.reshape((M, M)).T
            if M > N:
                yq = regroup(yq, M//N)
            Zq[r:r+M, c:c+M] = yq

    Zi = quant2(Zq, qstep, qstep)
    Zp = perform_Pr(Zi, C, s)
    return Zp

def eightLBTjpegtune(X):
    X = X-128.0
    SSIMscore = 0
    for k in range(120, 150, 2):
        step_size = 16
        calc_bits = 100000
        k = k/100
        while calc_bits > 40960:
            vlctemp, _ = jpegencLBT(X, step_size, s=k)
            calc_bits = vlctemp[:,1].sum()
            step_size += 1
        vlc, _ = jpegencLBT(X, step_size, k)
        Z = jpegdecLBT(vlc, step_size, k)
        rmserr = np.std(Z - X)
        scoretemp = visualerror(X, Z)
        print(scoretemp)
        if scoretemp > SSIMscore:
            SSIMscore = scoretemp
            Zfinal = Z
    return Zfinal, rmserr, calc_bits, SSIMscore


def eightDCTjpeg(X):
    X = X- 128.0
    step_size = 16
    calc_bits = 100000
    while calc_bits > 40960:
        vlctemp, _ = jpegenc(X, step_size)
        calc_bits = vlctemp[:,1].sum()
        step_size += 1
    vlc, _ = jpegenc(X, step_size)
    Z = jpegdec(vlc, step_size)
    rmserr = np.std(Z - X)
    Verr = visualerror(X, Z)
    return Z, rmserr, calc_bits, Verr
def compress(X):
    ZLBT, eLBT, bLBT, scoreLBT = eightLBTjpegtune(X)
    ZDCT, eDCT, bDCT, scoreDCT = eightDCTjpeg(X)
    if scoreDCT > scoreLBT:
        print(f"decoded with an rms error {eDCT} using {bDCT} bits and an SSIM score of {scoreDCT} (using DCT)")
        fig, ax = plt.subplots()
        plot_image(ZDCT, ax=ax)
    else:
        print(f"decoded with an rms error {eLBT} using {bLBT} bits and an SSIM score of {scoreLBT} (using LBT)")
        fig, ax = plt.subplots()
        plot_image(ZLBT, ax=ax)


def optimise()