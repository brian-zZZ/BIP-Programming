""" This file is the implementation of particle detection at pixel resolution.
    Author: Weihong Zhang, AI College, UCAS.
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from scipy.interpolate import interp2d
from scipy.ndimage import convolve

from ImplGaussianFilter import gaussian_kernel
from PixelParticleDetect import pixel_particle_detection



class miscs_utils:
    """ Miscellaneous utilities that provide auxiliary functions for all detection methods.
    """
    def __init__(self, im: np.ndarray):
        self.im = im
        self.im_interp = None

    def interpolation(self, ) -> np.ndarray:
        """ Interpolate the input 2D image array, return the Interpolated array.
        """
        height, width = self.im.shape
        interp_factor = 65 // 13 # 5 time

        # Define the interpolation function
        Y = np.arange(0, height)
        X = np.arange(0, width)
        f = interp2d(X, Y, self.im, kind='cubic')
        # Interpolate with the interpolation function
        Y_interp = np.linspace(0, height, height * interp_factor)
        X_interp = np.linspace(0, width, width * interp_factor)
        im_interp = f(X_interp, Y_interp)
        self.im_interp = im_interp

        return im_interp


    def visualize_save(self, particles_coords: list, save_base_path: str, im_idx: int):
        """ Visualize the detected particles by attaching them to the original image,
            then save the integrated detection figure.
        """
        plt.rcParams['font.family'] = ['Times New Roman']
        plt.rcParams.update({'font.size': 10})
        fig = plt.figure()
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
        plt.suptitle(f"Frame-{im_idx}")

        ax1.imshow(self.im_interp, cmap=plt.cm.gray)
        ax2.imshow(self.im_interp, cmap='Greys_r')

        # Scatter plot upon original image
        coords_y = [coord[0] for coord in particles_coords]
        coords_x = [coord[1] for coord in particles_coords]
        ax2.scatter(coords_x, coords_y, s=1.4, c='green', marker='*')

        plt.savefig(save_base_path+f'frame{im_idx}.tif', dpi=200)
        plt.show()


class gaussian_fit_detector:
    """ A 
    """
    def __init__(self, im: np.ndarray, im_interp: np.ndarray,
                 particles_coords: list[list], interp_factor: Optional[float] = None):
        self.im = im
        self.im_interp = im_interp
        self.particles_coords = particles_coords
        self.interp_factor = interp_factor if interp_factor is not None else 65 // 13


    def simplex_gaussian_kernel(self, h_shift: float, w_shift: float) -> np.ndarray:
        """ Generate simplex Gaussian kernel with specific shifts.
        """
        size = self.interp_factor # kernel size equals to interpolation factor
        A, B = 1, 1 # constants that don't contribute to the relative size of the final result
        hh, ww = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-(((hh - h_shift)**2 + (ww - w_shift)**2) / B)) * A
        return g / g.sum()


    def gaussian_fit_detect(self, ) -> list:
        """ Fit a 2D Gaussian using a simplex algorithm with a least-squares estimator.
            Return the sub-pixel particles coordinates.
        """
        subpixel_particles_coords = []
        
        for h, w in self.particles_coords:
            h_interp, w_interp = h * self.interp_factor, w * self.interp_factor # mapping
            diffs = [] # store the fitting difference for each shift
            # Shift from the interpolated origin point to find the minimized difference
            for h_shift in range(0, self.interp_factor+1):
                for w_shift in range(0, self.interp_factor+1):
                    kernel = self.simplex_gaussian_kernel(h_shift, w_shift)
                    diff = (np.abs(self.im[h, w] - kernel)).sum()
                    diffs.append([diff, [h_shift, w_shift]])
            print(kernel)

            diffs = pd.DataFrame(diffs, columns=['difference', 'shifts'])
            diffs.sort_values(by=['difference'], ascending=True, inplace=True)
            h_shift, w_shift = diffs['shifts'][0]
            print(h_shift, w_shift)
            subpixel_particle_coords = [h_interp + h_shift, w_interp + w_shift]
            subpixel_particles_coords.append(subpixel_particle_coords)

        return subpixel_particles_coords



def main():
    save_base_path = './processed_imgs/subpixel_particle_detection/'
    os.makedirs(save_base_path, exist_ok=True)

    rayleigh_radius = lambda _lambda, NA: (0.61 * _lambda / 100) / NA
    _lambda, NA = 515, 1.4
    sigma = rayleigh_radius(_lambda, NA) / 3
    smoothing_kernel = gaussian_kernel(sigma)

    im = cv2.imread('./images/001_a5_002_t001.tif', -1)
    im_filted = convolve(im, smoothing_kernel)
    particles_coords = pixel_particle_detection(im_filted, 3, 20)

    miscs = miscs_utils(im)
    im_interp = miscs.interpolation()

    interp_factor = 65 // 13
    gfit_detector = gaussian_fit_detector(im, im_interp, particles_coords, interp_factor)
    subpixel_particles_coords = gfit_detector.gaussian_fit_detect()
    print(particles_coords[:5], subpixel_particles_coords[:5])


    miscs.visualize_save(subpixel_particles_coords, save_base_path, 1)

    """
    im_interp = interpolation(im)
    fig = plt.figure()
    ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
    ax1.imshow(im, cmap=plt.cm.gray)
    ax2.imshow(im_interp, cmap='Greys_r')

    plt.show()
    """


if __name__ == '__main__':
    main()