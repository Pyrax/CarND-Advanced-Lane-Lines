import pickle
import cv2
import numpy as np


def undistort(image, coeff_fname):
    dist_pickle = None
    with open(coeff_fname, 'rb') as f:
        dist_pickle = pickle.load(f)
    mtx, dist = dist_pickle['mtx'], dist_pickle['dist']
    return cv2.undistort(image, mtx, dist, None, mtx)


class Undistorter:
    def __init__(self, coeff_fname, image_width, image_height):
        dist_pickle = None
        with open(coeff_fname, 'rb') as f:
            dist_pickle = pickle.load(f)

        if dist_pickle is not None:
            mtx, dist = dist_pickle['mtx'], dist_pickle['dist']
            # see: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
            self.map_x, self.map_y = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (image_width, image_height), 5)
        else:
            raise Exception('No pickle data for camera calibration loaded')

    def apply(self, image):
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def to_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def combine_binary(binary_image, second_binary):
    copy = np.zeros_like(binary_image)
    copy[(binary_image == 1) | (second_binary == 1)] = 1
    return copy


def color_threshold(image, channel=0, thresh=(0, 255)):
    image_channel = image[:, :, channel]
    binary = np.zeros_like(image_channel)
    binary[(image_channel > thresh[0]) & (image_channel <= thresh[1])] = 1
    return binary


class SobelGradientThresholder:
    def __init__(self, image, sobel_kernel=3):
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.sobel_kernel = sobel_kernel

    def abs_thresh(self, orient='x', thresh=(0, 255)):
        sobel = cv2.Sobel(self.gray, cv2.CV_64F, orient == 'x', orient == 'y', ksize=self.sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.amax(abs_sobel))

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    def mag_thresh(self, mag_thresh=(0, 255)):
        abs_sobelx = np.absolute(cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel))
        abs_sobely = np.absolute(cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel))
        mag = np.sqrt((abs_sobelx ** 2) + (abs_sobely ** 2))

        scaled_mag = np.uint8(255 * mag / np.amax(mag))
        binary_output = np.zeros_like(scaled_mag)
        binary_output[(scaled_mag > mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1
        return binary_output

    def dir_threshold(self, thresh=(0, np.pi/2)):
        abs_sobelx = np.absolute(cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel))
        abs_sobely = np.absolute(cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel))

        grad_dir = np.arctan2(abs_sobely, abs_sobelx)
        binary_output = np.zeros_like(grad_dir)
        binary_output[(grad_dir > thresh[0]) & (grad_dir <= thresh[1])] = 1
        return binary_output


def transform_perspective(image, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    image_size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def image_overlay(overlay, background_img, overlay_transparency=0.0):
    """
    Use an image as an overlay on another image that serves as background.
    overlay_transparency can be modified to make the overlay transparent so that the background
    is still visible (overlay_transparency must be between 0.0 and 1.0).

    Returns new image where both images have been combined.
    """
    overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(overlay_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Mask the overlay, so that it does not add black values to the background around
    # the overlay.
    background = cv2.bitwise_and(background_img, background_img, mask=mask_inv)
    foreground = cv2.bitwise_and(overlay, overlay, mask=mask)

    if overlay_transparency > 0.0 and overlay_transparency <= 1.0:
        background_mask = cv2.bitwise_and(background_img, background_img, mask=mask)
        foreground = weighted_img(foreground, background_mask, β=(1 - overlay_transparency))

    return cv2.add(foreground, background)
