import pickle
import cv2
import numpy as np


def undistort(image, coeff_fname):
    dist_pickle = None
    with open(coeff_fname, 'rb') as f:
        dist_pickle = pickle.load(f)
    mtx, dist = dist_pickle['mtx'], dist_pickle['dist']
    return cv2.undistort(image, mtx, dist, None, mtx)


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


class sobel_gradient_thresholder:
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
