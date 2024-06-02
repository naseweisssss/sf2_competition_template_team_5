""" This file contains the `decode` function. Feel free to split it into smaller functions """
import numpy as np
from cued_sf2_lab.jpeg import jpegdec

from .jpeg_dwt import jpegdwtdec
from .common import my_function, HeaderType, jpeg_quant_size, jpegdwt_quant_size, n_level

def decode(vlc: np.ndarray, header: HeaderType) -> np.ndarray:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
    """
    # replace this with your chosen decoding scheme

    hufftab, min_step = header
    Z =  jpegdwtdec(vlc, n_level, min_step, hufftab=hufftab, quantisation_scheme= 1, dcbits=8)
    Z += 128.0
    return Z