""" This file contains the `encode` function. Feel free to split it into smaller functions """
import numpy as np
from typing import Tuple
from cued_sf2_lab.jpeg import jpegenc
from jpeg_dwt import jpegdwtenc
from scipy.optimize import fsolve
from cued_sf2_lab.jpeg import HuffmanTable
from common import HeaderType, jpegdwt_quant_size, n


def header_bits(header: HeaderType) -> int:
    """ Estimate the number of bits in your header.
    
    If you have no header, return `0`. """
    # replace this with your size estimate, and a comment explaining how you got it!

    huffcode = header
    return (len(huffcode.bits) + len(huffcode.huffval)) * 8 + 64 #float size for min step


def encode(X: np.ndarray) -> Tuple[np.ndarray, HeaderType]:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
    """
    # replace this with your chosen encoding scheme. If you do not use a header,
    # then `return vlc, None`.
    X_normalised = X - 128.0
    min_step = fsolve(bit_diff, x0 = jpegdwt_quant_size, args = (X_normalised))[0]
    min_step = np.float64(min_step)
    vlc, hufftab = jpegdwtenc(X_normalised, n, min_step, opthuff=True, dcbits=8)
    return vlc, (hufftab, min_step)

def bit_diff(qstep: float, img: np.ndarray) -> float:
    target_bits = 40960 - 1424  - 64  # using opthuff, need to subtract 1424 to transmit table

    vlc, hufftab = jpegdwtenc(img, n, qstep, opthuff=True, dcbits= 9)
    total_bits = sum(vlc[:, 1])
    tbit_diff = np.abs(target_bits - total_bits)

    return tbit_diff
