from encoder import encode, header_bits
from decoder import decode
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.dct import colxfm, dct_ii, regroup
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.laplacian_pyramid import quant1, quant2, quantise, bpp
from cued_sf2_lab.jpeg import diagscan, runampl, huffdes, huffenc, huffgen, huffdflt, dwtgroup, HuffmanTable
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from scipy.optimize import fsolve, minimize
import numpy as np
import matplotlib.pyplot as plt


X1, _ = load_mat_img('images/lighthouse.mat', img_info='X')
X2, _ = load_mat_img('images/bridge.mat', img_info='X')
X3, _ = load_mat_img('images/flamingo.mat', img_info='X')
X4, _ = load_mat_img('images/SF2_competition_image_2019.mat', img_info='X')
X5, _ = load_mat_img('images/SF2_competition_image_2020.mat', img_info='X')
X6, _ = load_mat_img('images/SF2_competition_image_2021.mat', img_info='X')
X7, _ = load_mat_img('images/SF2_Competition_Image2022.mat', img_info='X')
X8, _ = load_mat_img('images/SF2_competition_image_2023.mat', img_info = 'X')


images_list = [X1, X2, X3, X4, X5, X6, X7, X8]

for i in range(len(images_list)):   
    vlc, hufftab = encode(images_list[i])

    print("This is " + str(i + 1))
    print("-" *30)
    n = 4
    suffecient_step = 1
    ref = bpp(quantise(images_list[i], 17))*images_list[i].size

    vlc, huff = encode(images_list[i])
    Z = decode(vlc, huff)
    fig, ax = plt.subplots()
    plot_image(Z, ax = ax)
    compression_ratio_mse = ref/(sum(vlc[:, 1]) + 1424)
    print("suffecient mse step: {}".format(suffecient_step))
    print(ref)
    print("No. of bits for mse: {}".format(sum(vlc[:, 1]) + 1424))
    print("RMS error for mse: {}".format(np.std(images_list[i] - Z)))
    print("Compression ratios for mse: {}".format(compression_ratio_mse))