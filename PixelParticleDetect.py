""" This file is the implementation of particle detection at pixel resolution.
    Author: Weihong Zhang, AI College, UCAS.
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from ImplGaussianFilter import gaussian_kernel



def calibration(im: np.ndarray) -> list:
    """ Crop a rectangular region in the image background area containing background noise.
        Return the calculated mean and standard deviation of the region.
        Notice: the coordinates below has been selected manually to ensure being a background.
    """
    height0, width0, scale = 30, 1200, 100
    im_cropped = im[height0: height0+scale, width0: width0+scale]
    mean, std = im_cropped.mean(), im_cropped.std()

    return [round(mean, 2), round(std, 2)]


def detect_extremums(im: np.ndarray, mask_scale: int) -> np.ndarray:
    """ Pixelwise extremums detection, return an indicative array with each pixel assigning a flag,
        where 0, 1, 2 represents local minima, normal point and local maxima respectively.
    """
    extremums_indicator = np.ones(im.shape).astype(np.int64)

    height, width = im.shape
    mask_half_scale = mask_scale // 2
    im = np.pad(im, mask_half_scale, mode='symmetric') # pad to yeild the same shape after masking
    
    patch_pixels_minus_one = np.math.pow(mask_scale, 2) - 1
    for row in range(mask_half_scale, height+mask_half_scale):
        for col in range(mask_half_scale, width+mask_half_scale):
            mask_patch = im[row-mask_half_scale: row+mask_half_scale+1,
                            col-mask_half_scale: col+mask_half_scale+1]
            if (im[row, col] > mask_patch).sum() == patch_pixels_minus_one:
                extremums_indicator[row-mask_half_scale, col-mask_half_scale] = 2
            elif (im[row, col] < mask_patch).sum() == patch_pixels_minus_one:
                extremums_indicator[row-mask_half_scale, col-mask_half_scale] = 0
    
    return extremums_indicator
            

def nearest_local_association(im: np.ndarray, extremums_indicator: np.ndarray) -> tuple[list]:
    """ Establish corresponind between an maxima and its nearset 4 minimas.
        Compared to the direct brute-force exhaustion that mentioned in the class, here is a
        efficient nearest neighbors searching strategy via expanding radius when is needed.
        
        Returns: a tuple of correspondence including intensities and coordinates:
                 ( [ [maxima_intensity, [minima_intensity x 4]] x n ],
                   [ [maxima_coordinates, [minima_coordinates x 4]] x n ] )
    """
    corresponding_intensities, corresponding_coords = [], []
    num_minimas_selected = 4 # select four nearest neighbors

    for h, w in np.argwhere(extremums_indicator == 2): # for each maxima
        radius = 1 # utilize squared approximation
        num_minimas_searched = 0 # already seached minimas for a given maxima
        # Loop until enough corresponding minimas are searched
        while num_minimas_searched < num_minimas_selected:
            # Search in a patch with a gradually expanding radius
            patch = im[h-radius: h+radius+1, w-radius: w+radius+1]
            patch_idt = extremums_indicator[h-radius: h+radius+1, w-radius: w+radius+1]
            patch_minimas_count = (patch_idt == 0).sum()
            
            if patch_minimas_count >= num_minimas_selected: # minimas within a patch satisfy
                distances = [] # store the euclidean distance for each minimas
                for h_min_rel, w_min_rel in np.argwhere(patch_idt == 0):
                    h_min_abs, w_min_abs = h_min_rel + h -1, w_min_rel + w -1
                    dist = np.math.sqrt(np.math.pow(h - h_min_abs, 2) + np.math.pow(w - w_min_abs, 2))
                    distances.append([[h_min_abs, w_min_abs], dist, patch[h_min_rel, w_min_rel]])
                
                # Sort by euclidean distance, then dice the nearest multiple minimas
                distances = pd.DataFrame(distances, columns=['minima_coord', 'distance', 'intensity'])
                distances.sort_values(by=['distance'], ascending=True, inplace=True)
                minimas_intensity = list(distances['intensity'][:num_minimas_selected])
                minimas_coord = list(distances['minima_coord'][:num_minimas_selected])
                corresponding_intensities.append([im[h, w], minimas_intensity])
                corresponding_coords.append([[h, w], minimas_coord])
                break
            else:
                # Update parameters
                num_minimas_searched = patch_minimas_count
                radius += 1

    return [corresponding_intensities, corresponding_coords]


def pixel_particle_detection(im: np.ndarray, mask_scale: int, quantile: float) -> list:
    """ Detect particle at pixel resolution for a single frame of image.
        Return the detected particle coordinates as the visualization handle.
    """
    particles_coords = []

    # Calibration of dark noise
    _, std_background = calibration(im)

    # Detect local extremums
    extremums_indicator = detect_extremums(im, mask_scale)

    # Efficiently establish correspondence
    corresponding_intensities, corresponding_coords = nearest_local_association(im, extremums_indicator)

    # Statistical selection of features
    std_delta = np.math.sqrt(0 + np.math.pow(std_background, 2) / 4.) # four neighbors
    for maxima_idx, (maxima_int, minimas_int) in enumerate(corresponding_intensities):
        # A maxima is sigificant if its intensity is greater than the mean of its background, i.e. the
        # nearest multiple minimas with a rather distinct confidence quantile, i.e. a critical interval
        if (maxima_int - np.mean(minimas_int)) > quantile * std_delta:
            particles_coords.append(corresponding_coords[maxima_idx][0])

    return particles_coords


def detection_visualize_save(im: np.ndarray, particles_coords: list,
                            save_base_path: str, im_idx: int):
    """ Visualize the detected particles by attaching them to the original image,
        then save the integrated detection figure.
    """
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure()
    ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
    plt.suptitle(f"Frame-{im_idx}")

    ax1.imshow(im, cmap=plt.cm.gray)
    ax2.imshow(im, cmap='Greys_r')

    # Scatter plot upon original image
    coords_y = [coord[0] for coord in particles_coords]
    coords_x = [coord[1] for coord in particles_coords]
    ax2.scatter(coords_x, coords_y, s=1.4, c='green', marker='*')

    plt.savefig(save_base_path+f'frame{im_idx}.tif', dpi=200)
    plt.show()


def main():
    save_base_path = './processed_imgs/pixel_particle_detection/'
    os.makedirs(save_base_path, exist_ok=True)

    # Config the gaussian kernel for smoothing filtering
    rayleigh_radius = lambda _lambda, NA: (0.61 * _lambda / 100) / NA
    _lambda, NA = 515, 1.4
    gaussian_std = rayleigh_radius(_lambda, NA) / 3
    kernel = gaussian_kernel(gaussian_std)

    # Extremums detection comparison upon different masks
    print("Extremums detection comparison upon 3x3 mask and 5x5 mask ...")
    im = cv2.imread('./images/001_a5_002_t001.tif', -1)
    im_filted = convolve(im, kernel)
    mask_scale1, mask_scale2 = 3, 5
    ext_idt1 = detect_extremums(im_filted, mask_scale1)
    ext_idt2 = detect_extremums(im_filted, mask_scale2)
    print('Counts: maximas | others | minimas')
    print(f"3x3 -> { (ext_idt1==2).sum()} | {(ext_idt1==1).sum()} | {(ext_idt1==0).sum()}")
    print(f"5x5 ->  {(ext_idt2==2).sum()} | {(ext_idt2==1).sum()} | {(ext_idt2==0).sum()}")

    # Particle detection
    print("Start detecting particles")
    Q = input("Please specify confidence quantile first (20 by default): ")
    for i in range(1, 6):
        im = cv2.imread(f'./images/001_a5_002_t00{i}.tif', -1) # 16bit
        im_filted = convolve(im, kernel)
        print(f"Detecting frame-{i} ...")
        particles_coords = pixel_particle_detection(im_filted, mask_scale1, float(Q))
        detection_visualize_save(im, particles_coords, save_base_path, i)



if __name__ == '__main__':
    main()
