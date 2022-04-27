""" This file is the implementation of image intensity derivatives calculation.
    Author: Weihong Zhang, AI College, UCAS.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from ImplGaussianFilter import gaussian_kernel



def shift(array: np.ndarray, shift_spec: list or tuple) -> np.ndarray:
    """ Shift in a specified direction for approximate derivatives calculation.
    """
    padded = np.pad(array, 1, mode='edge')
    y, x = shift_spec
    shifted = padded[1 + y: -1 + y if y != 1 else None,
                     1 + x: -1 + x if x != 1 else None]
    return shifted


def derivatives(array: np.ndarray) -> list:
    """ Calculates the first order approximate derivatives for an array.
    """
    a = array
    dy = (shift(a, [1, 0]) - shift(a, [-1, 0])) / 2
    dx = (shift(a, [0, 1]) - shift(a, [0, -1])) / 2

    return [dy, dx]


def derivative_gaussian_conv(sigma: float, im: np.ndarray) -> list:
    """ Convolved with derivative Gaussian kernel,
        return a list of vertical and horizontal convolution result.
        Notice: Convolution first and then derivation is equivalent to
                derivation first and then convolution. Here is the later implementation.
    """
    # Generate Gaussian kernel
    kernel = gaussian_kernel(sigma)
    # Calculate both vertical and horizontal derivatives of kernel
    vert_deriv_kernel, hori_deriv_kernel = derivatives(kernel)
    # Convolution operation
    vert_deriv_filted = convolve(im, vert_deriv_kernel)
    hori_deriv_filted = convolve(im, hori_deriv_kernel)

    return [vert_deriv_filted, hori_deriv_filted]


def save_show_NarrowWide(orig_im: np.ndarray, vert_deriv_filted_im: np.ndarray,
                         hori_deriv_filted_im: np.ndarray, sigma: float, save_base_path: str):
    """ Integrate original, vertical and horizontal derivative images, the save and show.
        Notice: to ensure visualization quality, this method plot in shape (3, 1) for narrow and wide image.
    """
    # Integrate all the three images
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    plt.figure(dpi=120)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5) # adjust height seperation for better visualization
    plt.suptitle(f'Sigma {sigma:.1f}', fontsize=12)
    plt.subplot(3, 1, 1)
    plt.title("Original")
    plt.imshow(orig_im, cmap=plt.cm.gray)
    plt.subplot(3, 1, 2)
    plt.title("Vertical")
    plt.imshow(vert_deriv_filted_im, cmap='gray')
    plt.subplot(3, 1, 3)
    plt.title("Horizontal")
    plt.imshow(hori_deriv_filted_im, cmap='gray')
    
    # Save the integrated image
    integrated_im = plt.gcf()
    integrated_im.savefig(save_base_path+f'ax2_sigma{sigma}.tif')
    # Show the integrated image
    plt.show()

def save_show_Sqaure(orig_im: np.ndarray, vert_deriv_filted_im: np.ndarray,
                     hori_deriv_filted_im: np.ndarray, sigma: float, save_base_path: str):
    """ Act like 'save_show_NarrowWide' above with a slightly change.
        Notice: to ensure visualization quality, this method plot in shape (2, 2) for square image.
    """
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    plt.figure(dpi=120)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'Sigma {sigma:.1f}', fontsize=12)
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(orig_im, cmap=plt.cm.gray)
    plt.subplot(2, 2, 2)
    plt.title("Vertical")
    plt.imshow(vert_deriv_filted_im, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title("Horizontal")
    plt.imshow(hori_deriv_filted_im, cmap='gray')
    plt.savefig(save_base_path+f'cell_sigma{sigma}.tif', dpi=200) # set higher dpi to distinguish noises
    plt.show()


def main():
    ax2_im = cv2.imread('./images/axon02.tif', -1)
    cell_im = cv2.imread('./images/cell_nucleus.tif', -1)
    sigmas = [1., 2., 5.]
    save_base_path = './processed_imgs/derivative_gaussian/'
    os.makedirs(save_base_path, exist_ok=True)

    for sigma in sigmas:
        ax2_vert_deriv_filted, ax2_hori_deriv_filted = derivative_gaussian_conv(sigma, ax2_im)
        save_show_NarrowWide(ax2_im, ax2_vert_deriv_filted, ax2_hori_deriv_filted, sigma, save_base_path)

        cell_vert_deriv_filted, cell_hori_deriv_filted = derivative_gaussian_conv(sigma, cell_im)
        save_show_Sqaure(cell_im, cell_vert_deriv_filted, cell_hori_deriv_filted, sigma, save_base_path)


if __name__ == '__main__':
    main()
