""" This file contains the `decode` function. Feel free to split it into smaller functions """
import numpy as np
from cued_sf2_lab.jpeg import jpegdec
from typing import Tuple, Any


from jpeg_dwt import jpeg2000dec
# from common import my_function, HeaderType, jpeg_quant_size, jpeg2000_quant_size, n_level
from cued_sf2_lab.jpeg import HuffmanTable
HeaderType = Tuple[HuffmanTable, float]

n_level = 4
jpeg2000_quant_size = 18.03030303030303
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
    print("fdddfdff")
    Z =  jpeg2000dec(vlc, 4, min_step, hufftab=hufftab, quantisation_scheme= 1, dcbits=8)
    Z += 128.0
    return Z