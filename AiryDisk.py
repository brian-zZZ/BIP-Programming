""" This file is the implementation of both Airy disk plotter 
    as well as the fitting of the Airy disk using a Gaussian kernel
    Author: Weihong Zhang, AI College, UCAS.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spl



def airy_disk(_lambda: int, NA: float) -> np.ndarray:
    """ PSF (point spread function), with the shape of so-called airy disk.
    """
    _lambda /= 100. # rescale to normalize the peak magnitude to 1
    r = np.linspace(-10, 10, 10000)
    a = (2 * np.pi * NA) / _lambda
    h = np.power(2 * spl.jv(1, a*r) / (a * r), 2)
    return h


def gaussian_distribution(sigma: float) -> np.ndarray:
    """ Gaussian distribution function.
    """ 
    x = np.linspace(-10, 10, 10000)
    y = np.exp(-np.power(x, 2) / (2 * np.math.pow(sigma, 2))) / np.math.sqrt(2 * np.pi * sigma)
    return y


def grid_search_sigma(h: np.ndarray) -> float:
    """ Grid search to find the best sigma of a gaussian kernel that minimize the
        sum of squared difference between airy disk and guassian kernel.
    """
    sigma_grids = np.linspace(.001, 3.001, 3000) # as a posterior knowledge, best sigma lies wihthin [0, 3]
    best_sigma, min_diff = .0, np.inf
    for sigma in sigma_grids:
        g = gaussian_distribution(sigma)
        diff = np.power(h - g, 2).sum()
        # Update if a smaller difference is encountered
        if diff < min_diff:
            best_sigma = sigma
            min_diff = diff

    return best_sigma


def get_radius(h: np.ndarray) -> float:
    """ Calculate radius, i.e. the first local minima of an airy disk.
    """
    # Calculate the first approximate derivative
    h = np.pad(h[len(h)//2:], 1, mode='edge') # the half on the right side
    shift = lambda m: h[1 + m: -1 + m if m != 1 else None]
    first_derivative = (shift(1) - shift(-1)) / 2

    # The first place where the sign of the values jumps is the first local minima
    radius_idx = np.where(first_derivative>.0)[0][0]
    radius = (radius_idx / 5000) * 10

    return radius


def main():
    var_list = [[480, 0.5], [520, 0.5], [680, 0.5],
                [520, 1.0], [520, 1.4], [680, 1.5]]
    r = np.linspace(-10, 10, 10000)
    palette = plt.get_cmap('Set1')
    save_base_path = './processed_imgs/airy_disk/'
    os.makedirs(save_base_path, exist_ok=True)

    plt.figure()
    plt.title("Airy disk")
    for i, var_config in enumerate(var_list):
        print("=== " * 8, "\nConfiguration: ", var_config)
        h = airy_disk(*var_config)
        radius = get_radius(h)
        print("The radius of Airy disk is {:.3f}".format(float(radius)))
        plt.plot(r, h, '-', color=palette(i), label=f'{var_config}')
        
        sigma = grid_search_sigma(h)
        print(f"The best fit sigma is %.3f" % sigma)
        radius_over_sigma = radius / sigma # conclusion: a constant approximately equals to 2.90
        print(f"Radius / sigma = {radius_over_sigma:.2f}")
        # g = gaussian_distribution(sigma)
        # plt.plot(r, g, '--', color=palette(i), label=f'sigma: {sigma:.2f}')
    
    plt.grid()
    plt.legend()
    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')
    plt.savefig(save_base_path+'airy_disk_comp.tif')
    plt.show()


if __name__ == '__main__':
    main()