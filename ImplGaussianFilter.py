""" This file is the implementation of Gaussian filter.
    Author: Weihong Zhang, AI College, UCAS.
"""
import os
import cv2
import numpy as np
from scipy.ndimage import convolve # support 16bit images



def gaussian_kernel(sigma: float) -> np.ndarray:
    """ Generate Gaussian kernel with specific sigma.
        Assume sigma_x equals to simgma_y, i.e., both are sigma.
    """
    size = 2 * np.ceil(3 * sigma) + 1 # 3σ and centroid
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) / (2 * np.pi * sigma**2)
    return g / g.sum() # normalize the filter template


def convolve_save(sigma: float, im_array: np.ndarray, save_base_path: str) -> np.ndarray:
    """ Gaussian convolution then save the filtered image locally.
    """
    print("Gaussian filtering with sigma=", sigma)
    # Generate Gaussian kernel with specific sigma configuration
    kernel = gaussian_kernel(sigma)
    # Convolve with Gaussian kernel
    im_filted = convolve(im_array, kernel)

    cv2.imwrite(save_base_path+f'axon01_sigma{sigma:.1f}.tif', im_filted)


def main():
    im = cv2.imread('./images/axon01.tif', -1) # 16bit
    sigmas = [1., 2., 5., 7.]
    save_base_path = './processed_imgs/gaussian_filted/'
    os.makedirs(save_base_path, exist_ok=True)

    for sigma in sigmas:
        convolve_save(sigma, im, save_base_path)

    in_var = input("Please input another sigma (float >0.0). ")
    in_var = float(in_var)
    assert in_var > 0.0, \
           f"The value of sigma must greater than 0.0, {in_var} is given."
    convolve_save(in_var, im, save_base_path)



if __name__ == '__main__':
    main()
