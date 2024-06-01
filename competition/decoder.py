""" This file contains the `decode` function. Feel free to split it into smaller functions """
import numpy as np
from cued_sf2_lab.jpeg import jpegdec
from typing import Tuple, Any


from jpeg_dwt import jpegdwtdec
# from common import my_function, HeaderType, jpeg_quant_size, jpeg2000_quant_size, n_level
from cued_sf2_lab.jpeg import HuffmanTable
from common import HeaderType, n, jpegdwt_quant_size, jpeg_quant_size


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
    Z =  jpegdwtdec(vlc, 4, min_step, hufftab=hufftab, dcbits=8)
    Z += 128.0
    return Z