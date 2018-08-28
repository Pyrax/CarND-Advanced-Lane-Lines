import numpy as np
import cv2
import ImageFunctions as imgf
import LaneDetector as ld
import utils


class Pipeline:
    def __init__(self, camera_calibration_file, **kwargs):
        self.camera_calibration_file = camera_calibration_file
        self.undistorter = None # will be set with first processed image because it needs image size
        self.lane_det = ld.LaneDetector(**kwargs)

    def execute(self, image):
        undistorted_image, binary_warped = self.preprocess_image(image)
        self.lane_det.run_on_image(binary_warped)

        lane_area = self.lane_det.get_lane_area()
        colored_lanes = self.lane_det.get_lanes()

        trans_src, trans_dst = self.get_transform_params(image)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        lane_area_unwarped = imgf.transform_perspective(lane_area, trans_dst, trans_src)
        lanes_unwarped = imgf.transform_perspective(colored_lanes, trans_dst, trans_src)

        image_with_area = cv2.addWeighted(undistorted_image, 1., lane_area_unwarped, .3, 1)
        return imgf.image_overlay(lanes_unwarped, image_with_area, overlay_transparency=.1)

    def evaluate(self, image):
        undistorted_image, binary_warped = self.preprocess_image(image)
        self.lane_det.run_on_image(binary_warped)
        return self.lane_det.get_visualization()

    def preprocess_image(self, image):
        if self.undistorter is None:
            self.undistorter = imgf.Undistorter(self.camera_calibration_file, image.shape[1], image.shape[0])
        undistorted_image = self.undistorter.apply(image)

        hls_image = imgf.to_hls(undistorted_image)
        gray_image = imgf.to_grayscale(undistorted_image)

        s_threshold = imgf.color_threshold(hls_image, 2, (170, 255))
        gray_threshold = imgf.color_threshold((np.dstack((gray_image, gray_image, gray_image))), 0, (200, 255))

        combined_color_thresholds = imgf.combine_binary(s_threshold, gray_threshold)

        sobel = imgf.SobelGradientThresholder(undistorted_image, sobel_kernel=3)
        gradient_thresholds = sobel.abs_thresh(orient='x', thresh=(20, 100))

        thresholded_image = imgf.combine_binary(gradient_thresholds, combined_color_thresholds)

        trans_src, trans_dst = self.get_transform_params(image)
        binary_warped = imgf.transform_perspective(thresholded_image, trans_src, trans_dst)
        return undistorted_image, binary_warped

    def get_transform_params(self, image):
        image_size = image.shape[1], image.shape[0]

        top_y = image_size[1] / 1.6
        bottom_y = image_size[1]

        bottom_x_margin = image_size[0] / 8
        bottom_left_x = 50 + bottom_x_margin
        bottom_right_x = image_size[0] - bottom_x_margin

        top_x_margin = image_size[0] / 2.18
        top_left_x = top_x_margin + 6
        top_right_x = image_size[0] - top_x_margin - 2

        # Having the lines one fourth away from left and right corner of the transformed image seems optimal
        # because that way the lanes are centered in the left and right half at the bottom of the image.
        dst_left_x = image_size[0] / 4
        dst_right_x = dst_left_x * 3

        trans_src = np.float32([
            [top_left_x, top_y],        # top-left
            [top_right_x, top_y],       # top-right
            [bottom_right_x, bottom_y], # bottom-right
            [bottom_left_x, bottom_y],  # bottom-left
        ])
        trans_dst = np.float32([
            [dst_left_x, 0],                # top-left
            [dst_right_x, 0],               # top-right
            [dst_right_x, image_size[1]],   # bottom-right
            [dst_left_x, image_size[1]]     # bottom-left
        ])

        return trans_src, trans_dst
