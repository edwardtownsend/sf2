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

def jpegenc_dct_lbt(X, step_table, C, s=None, N=8, M=8, opthuff=False, dcbits=8):
    if M % N != 0:
        raise ValueError('M must be an integer multiple of N!')

    ### NEW CODE
    Y = forward_dct_lbt(X, C, s)
    Yq = quant1_jpeg(Y, step_table).astype('int')
    ###

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

def jpegdec_dct_lbt(vlc, step_table, C, s=None, N=8, M=8, hufftab=None, dcbits=8, W=256, H=256):
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

    ### NEW CODE
    Zi = quant2_jpeg(Zq, step_table)
    Zp = inverse_dct_lbt(Zi, C, s)
    ###
    
    return Zp

def jpegenc_dwt(X, num_levels, N=8, M=8, opthuff=False, dcbits=8):
    if M % N != 0:
        raise ValueError('M must be an integer multiple of N!')

    ### NEW CODE
    Yq_dwt = gen_Y_dwt_equal_mse(X, num_levels)[0]
    Yq = dwtgroup(Yq_dwt, num_levels)
    ###

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

            ### NEW/MODIFIED CODE
            dccoef = yqflat[0]
            max_dccoef = np.max(np.abs(Yq))
            dccoef_normalized = int((dccoef / max_dccoef) * (2 ** (dcbits - 1)))
            dccoef_final = dccoef_normalized + 2 ** (dcbits - 1)
            
            if dccoef_final not in range(2**dcbits):
                raise ValueError('DC coefficients too large for desired number of bits')
            vlc.append(np.array([[dccoef_final, dcbits]]))
            ###

            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))

    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    if opthuff:
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

                ### NEW CODE
                dccoef = yqflat[0]
                dccoef_normalized = int((dccoef / max_dccoef) * (2 ** (dcbits - 1)))
                dccoef_final = dccoef_normalized + 2 ** (dcbits - 1)
                vlc.append(np.array([[dccoef_final, dcbits]]))
                ###

                ra1 = runampl(yqflat[scan])
                vlc.append(huffenc(huffhist, ra1, ehuf))
                
        vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    return vlc, dhufftab