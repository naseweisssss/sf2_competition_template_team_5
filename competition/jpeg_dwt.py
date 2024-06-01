from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, NamedTuple, Optional
from cued_sf2_lab.dct import colxfm, dct_ii, regroup
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.laplacian_pyramid import quant1, quant2, quantise, bpp
from cued_sf2_lab.jpeg import diagscan, runampl, huffdes, huffenc, huffgen, huffdflt, dwtgroup, HuffmanTable
from cued_sf2_lab.familiarisation import load_mat_img, plot_image

def nlevdwt(X, n):
    m = X.shape[0]
    Y = X.copy()
    for _ in range(n):
        Y[:m, :m] = dwt(Y[:m, :m])
        m = m//2
    return Y

def nlevidwt(Y, n):
    m = Y.shape[0]//2**(n-1)
    Z = Y.copy()
    for _ in range(n):
        Z[:m , :m] = idwt(Z[:m, :m])
        m = m*2
    return Z

def quant1dwt(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = dwtstep.shape[1] - 1
    m = Y.shape[0]
    Yq = Y.copy()
    for l in range(n):
        Yq[0:m//2, m//2: m] = quant1(Yq[0:m//2, m//2:m], dwtstep[0, l], dwtstep[0, l])
        
        Yq[m//2: m, 0:m//2] = quant1(Yq[m//2: m, 0:m//2], dwtstep[1, l], dwtstep[1, l])
        
        Yq[m//2: m, m//2: m] = quant1(Yq[m//2: m, m//2: m], dwtstep[2, l], dwtstep[2, l])
        m = m//2
                          
    Yq[0:m, 0:m] = quant1(Yq[0:m, 0:m], dwtstep[0,n], dwtstep[0,n])
    
    return Yq

def quant2dwt(Z: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = dwtstep.shape[1] - 1
    m = Z.shape[0]
    Zq = Z.copy()
    for l in range(n):
        Zq[0:m//2, m//2: m] = quant2(Zq[0:m//2, m//2:m], dwtstep[0, l], dwtstep[0, l])
        
        Zq[m//2: m, 0:m//2] = quant2(Zq[m//2: m, 0:m//2], dwtstep[1, l], dwtstep[1, l])
        
        Zq[m//2: m, m//2: m] = quant2(Zq[m//2: m, m//2: m], dwtstep[2, l], dwtstep[2, l])
        m = m//2
                          
    Zq[0:m, 0:m] = quant2(Zq[0:m, 0:m], dwtstep[0,n], dwtstep[0,n])
    
    return Zq

def step_ratios(n):
    '''
    Generate a 3 x (n + 1) matrix that stores the ratio of step sizes for sub-images with equal MSE

    Parameters:
        n: Levels of DWT
    
    Returns:
        step ratios: 3 x (n+1) matrix of ratio of step sizes normalised based on first input
    '''
    X_test = np.zeros((256, 256))
    dwt_ratios = np.ones((3, n + 1))
    m = 256
    for i in range(1, n + 1):
        Ytr = nlevdwt(X_test, i)
        Ytr[0:m//2, m//2: m][m//4][m//4] = 100
        Ztr = nlevidwt(Ytr, i)
        dwt_ratios[0][i-1] = np.sum(Ztr**2.0)
        
        Ybl = nlevdwt(X_test, i)
        Ybl[m//2: m, 0:m//2][m//4][m//4] = 100
        Zbl = nlevidwt(Ybl, i)
        dwt_ratios[1][i-1] = np.sum(Zbl**2.0)
        
        Ybr = nlevdwt(X_test, i)
        Ybr[m//2: m, m//2: m][m//4][m//4] = 100
        Zbr = nlevidwt(Ybr, i)
        dwt_ratios[2][i-1] = np.sum(Zbr**2.0)
        m = m//2
    Ytr = nlevdwt(X_test, n)
    Ytr[0:m, 0:m][m//2][m//2] = 100
    Ztr = nlevidwt(Ytr, n)
    dwt_ratios[0][n] = np.sum(Ztr**2.0)
    
    dwt_ratios = 1/np.sqrt(dwt_ratios)
    dwt_ratios = dwt_ratios/dwt_ratios[0][0]
    dwt_ratios[-1][-1] = 0
    dwt_ratios[-2][-1] = 0
    return dwt_ratios

def frequency_dependent_quantisation(n):
    k = 0.99
    dwt_ratios = np.zeros((3, n + 1))
    for i in range(n):
        dwt_ratios[0][i] = k**(i)
        dwt_ratios[1][i] = k**(i)
        dwt_ratios[2][i] = k**(i)
    
    dwt_ratios[0][n] = k**(n)

    return dwt_ratios

def jpeg2000enc(X: np.ndarray, n: float, qstep: float, quantisation_scheme: int, dcbits: int = 8,
        opthuff: bool = False, log: bool = False
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Encodes the image in X to generate a variable length bit stream.

    Parameters:
        X: the input greyscale image
        n: levels of DWT
        qstep: the quantisation step to use in encoding
        quantisation_scheme: choose between 0-const step, 1-equal MSE, 2-frequency dependent
        opthuff: if true, the Huffman table is optimised based on the data in X

    Returns:
        vlc: variable length output codes, where ``vlc[:,0]`` are the codes and
            ``vlc[:,1]`` the number of corresponding valid bits, so that
            ``sum(vlc[:,1])`` gives the total number of bits in the image
        hufftab: optional outputs containing the Huffman encoding
            used in compression when `opthuff` is ``True``.
    '''


    # DCT on input image X.
    N = 2**n
    if log:
        print('DWT of {} level'.format(n))
    Y = nlevdwt(X, n)

    # Quantise to integers.
    if log:
        print('Quantising to step size of {}'.format(qstep))

    if quantisation_scheme == 0:
        Yq = quant1(Y, qstep, qstep).astype('int')
    elif quantisation_scheme == 1:
        Yq  =  quant1dwt(Y, qstep*step_ratios(n)).astype('int') #for equal MSE?
    elif quantisation_scheme == 2:
        Yq = quant1dwt(Y, qstep*frequency_dependent_quantisation(n)).astype('int')
        
    Yr = dwtgroup(Yq, n)
    # Generate zig-zag scan of AC coefs.
    scan = diagscan(N)

    # On the first pass use default huffman tables.
    if log:
        print('Generating huffcode and ehuf using default tables')
    dhufftab = huffdflt(1)  # Default tables.
    huffcode, ehuf = huffgen(dhufftab)

    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.
    if log:
        print('Coding rows')
    sy = Yq.shape
    huffhist = np.zeros(16 ** 2)
    vlc = []
    # dcbits = 0
    # for r in range(0, sy[0], N):
    #     for c in range(0, sy[1], N):
    #         yr = Yr[r:r+N,c:c+N]
    #         # Possibly regroup
    #         yrflat = yr.flatten('F')
    #         # Encode DC coefficient first
    #         top_left = yrflat[0]
    #         if top_left ==0:
    #             continue
    #         if np.log2(np.abs(top_left)) == int(np.log2(np.abs(top_left))):
    #             dcbits = max(dcbits, int(np.ceil(np.log2(np.abs(top_left)) + 2)))
    #         else:
    #             dcbits = max(dcbits, int(np.ceil(np.log2(np.abs(top_left)) + 1)))

    for r in range(0, sy[0], N):
        for c in range(0, sy[1], N):
            yr = Yr[r:r+N,c:c+N]
            # Possibly regroup
            yrflat = yr.flatten('F')
            # Encode DC coefficient first
            dccoef = yrflat[0] + 2 ** (dcbits-1)
            if dccoef not in range(2**dcbits):
                print(dcbits, yrflat[0])
                raise ValueError(
                    'DC coefficients too large for desired number of bits')
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yrflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    # Return here if the default tables are sufficient, otherwise repeat the
    # encoding process using the custom designed huffman tables.
    if not opthuff:
        if log:
            print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        return vlc, dhufftab

    # Design custom huffman tables.
    if log:
        print('Generating huffcode and ehuf using custom tables')
    dhufftab = huffdes(huffhist)
    huffcode, ehuf = huffgen(dhufftab)

    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.
    if log:
        print('Coding rows (second pass)')
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], N):
        for c in range(0, sy[1], N):
            yr = Yr[r:r+N, c:c+N]
            # Possibly regroup
            yrflat = yr.flatten('F')
            # Encode DC coefficient first
            dccoef = yrflat[0] + 2 ** (dcbits-1)
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yrflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    if log:
        print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        print('Bits for huffman table = {}'.format(
            (16 + max(dhufftab.huffval.shape))*8))

    return vlc, dhufftab

def jpeg2000dec(vlc: np.ndarray, n: int, qstep: float, quantisation_scheme: int, dcbits: int = 8,
        hufftab: Optional[HuffmanTable] = None, 
        W: int = 256, H: int = 256, log: bool = False
        ) -> np.ndarray:
    '''
    Decodes a (simplified) JPEG bit stream to an image

    Parameters:

        vlc: variable length output code from jpegenc
        qstep: quantisation step to use in decoding
        n: levels of DWT
        quantisation_scheme: choose between 0-const step, 1-equal MSE, 2-frequency dependent
        hufftab: if supplied, these will be used in Huffman decoding
            of the data, otherwise default tables are used
        W, H: the size of the image (defaults to 256 x 256)

    Returns:

        Z: the output greyscale image
    '''

    opthuff = (hufftab is not None)
    N = 2**n

    # Set up standard scan sequence
    scan = diagscan(N)

    if opthuff:
        if len(hufftab.bits.shape) != 1:
            raise ValueError('bits.shape must be (len(bits),)')
        if log:
            print('Generating huffcode and ehuf using custom tables')
    else:
        if log:
            print('Generating huffcode and ehuf using default tables')
        hufftab = huffdflt(1)
    # Define starting addresses of each new code length in huffcode.
    # 0-based indexing instead of 1
    huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
    # Set up huffman coding arrays.
    huffcode, ehuf = huffgen(hufftab)

    # Define array of powers of 2 from 1 to 2^16.
    k = 2 ** np.arange(17)

    # For each block in the image:

    # Decode the dc coef (a fixed-length word)
    # Look for any 15/0 code words.
    # Choose alternate code words to be decoded (excluding 15/0 ones).
    # and mark these with vector t until the next 0/0 EOB code is found.
    # Decode all the t huffman codes, and the t+1 amplitude codes.

    eob = ehuf[0]
    run16 = ehuf[15 * 16]
    i = 0
    Zq = np.zeros((H, W))

    if log:
        print('Decoding rows')
    for r in range(0, H, N):
        for c in range(0, W, N):
            yq = np.zeros(N**2)

            # Decode DC coef - assume no of bits is correctly given in vlc table.
            cf = 0
            if vlc[i, 1] != dcbits:
                raise ValueError(
                    'The bits for the DC coefficient does not agree with vlc table')
            yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)

            i += 1

            # Loop for each non-zero AC coef.
            while np.any(vlc[i] != eob):
                run = 0

                # Decode any runs of 16 zeros first.
                while np.all(vlc[i] == run16):
                    run += 16
                    i += 1

                # Decode run and size (in bits) of AC coef.
                start = huffstart[vlc[i, 1] - 1]
                res = hufftab.huffval[start + vlc[i, 0] - huffcode[start]]
                run += res // 16
                cf += run + 1
                si = res % 16
                i += 1

                # Decode amplitude of AC coef.
                if vlc[i, 1] != si:
                    raise ValueError(
                        'Problem with decoding .. you might be using the wrong hufftab table')
                ampl = vlc[i, 0]

                # Adjust ampl for negative coef (i.e. MSB = 0).
                thr = k[si - 1]
                yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)

                i += 1

            # End-of-block detected, save block.
            i += 1

            yq = yq.reshape((N, N)).T

            Zq[r:r+N, c:c+N] = yq

    if log:
        print('Inverse quantising to step size of {}'.format(qstep))

    Zr = dwtgroup(Zq, -n)


    if quantisation_scheme == 0:
        Zi = quant2dwt(Zr, qstep*frequency_dependent_quantisation(n))
    elif quantisation_scheme == 1:
        Zi = quant2dwt(Zr, qstep*step_ratios(n)) #for equal MSE?
    elif quantisation_scheme == 2:
        Zi = quant2(Zr, qstep, qstep)
    if log:
        print('Inverse {} level DWT\n'.format(n))
    
    

    Z = nlevidwt(Zi, n)
    return Z


if __name__ == "__main__":
    # load in the image
    from cued_sf2_lab.familiarisation import load_mat_img, plot_image

    X1, _ = load_mat_img('lighthouse.mat', img_info='X')
    X2, _ = load_mat_img('bridge.mat', img_info='X')
    X3, _ = load_mat_img('flamingo.mat', img_info='X')
    X4, _ = load_mat_img('SF2_competition_image_2019.mat', img_info='X')
    X5, _ = load_mat_img('SF2_competition_image_2020.mat', img_info='X')
    X6, _ = load_mat_img('SF2_competition_image_2021.mat', img_info='X')
    X7, _ = load_mat_img('SF2_Competition_Image2022.mat', img_info='X')


    images = [X1, X2, X3, X4, X5, X6, X7]
    image_dec = []

    for i in range(len(images)):
        X = images[i]
        print(i)
        vlc, hufftab = encode(X)
        Z = decode(vlc, hufftab)
        image_dec.append(Z)

    n = 4
    suffecient_step = 1
    ref = bpp(quantise(X, 17))*X.size

    vlc_suff, huff_suff = jpeg2000enc(X, n, suffecient_step, opthuff= True, quantisation_scheme=1)
    Z_suff = jpeg2000dec(vlc_suff, n, suffecient_step, hufftab=huff_suff, quantisation_scheme=1)
    compression_ratio_mse = ref/(sum(vlc_suff[:, 1]) + 1424)
    print("suffecient mse step: {}".format(suffecient_step))

    print("No. of bits for mse: {}".format(sum(vlc_suff[:, 1]) + 1424))
    print("RMS error for mse: {}".format(np.std(X - Z_suff)))
    print("Compression ratios for mse: {}".format(compression_ratio_mse))