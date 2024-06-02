""" This file contains the `encode` function. Feel free to split it into smaller functions """
import numpy as np
from typing import Tuple, Any
from cued_sf2_lab.jpeg import jpegenc
from .jpeg_dwt import jpegdwtenc
from .common import HeaderType, jpeg_quant_size, jpegdwt_quant_size, n_level
from scipy.optimize import fsolve, minimize


n_level = 4
jpegdwt_quant_size = 19.161616161616
def header_bits(header: HeaderType) -> int:
    """ Estimate the number of bits in your header.
    
    If you have no header, return `0`. """
    # replace this with your size estimate, and a comment explaining how you got it!

    huffcode, min_step = header
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
    vlc, hufftab = jpegdwtenc(X_normalised, n_level, min_step, opthuff=True, quantisation_scheme=1, dcbits=8)
    return vlc, (hufftab, min_step)

def bit_diff(qstep: float, img: np.ndarray) -> float:
    target_bits = 40960 - 1424  - 64  # using opthuff, need to subtract 1424 to transmit table

    vlc, hufftab = jpegdwtenc(img, n_level, qstep, opthuff=True, quantisation_scheme=1, dcbits= 9)
    total_bits = sum(vlc[:, 1])
    tbit_diff = np.abs(target_bits - total_bits)

    return tbit_diff
