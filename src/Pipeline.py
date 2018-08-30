import numpy as np
import cv2
import ImageFunctions as imgf
import LaneDetector as ld


class Pipeline:
    def __init__(self, camera_calibration_file, transform_params):
        self.camera_calibration_file = camera_calibration_file

        self.M = cv2.getPerspectiveTransform(transform_params[0], transform_params[1])
        self.M_inv = cv2.getPerspectiveTransform(transform_params[1], transform_params[0])
        self.undistorter = None     # will be set with first processed image because it needs image size

    def execute(self, image):
        if self.undistorter is None:
            self.undistorter = imgf.Undistorter(self.camera_calibration_file, image.shape[1], image.shape[0])
        return self.create_frame_pipeline(image).execute()

    def evaluate(self, image):
        return self.create_frame_pipeline(image).evaluate()

    def create_frame_pipeline(self, image):
        return FramePipeline(image, self.undistorter, self.M, self.M_inv,
                             n_windows=9, margin=80, min_pix=40, poly_margin=50)


class FramePipeline:
    def __init__(self, input_image, undistorter, transform_M, transform_M_inv, **kwargs):
        self.input_image = input_image
        self.undistorter = undistorter
        self.transform_M = transform_M
        self.transform_M_inv = transform_M_inv
        self.lane_det = ld.LaneDetector(**kwargs)

    def execute(self):
        undistorted_image, binary_warped = self.preprocess_image(self.input_image)

        self.lane_det.run_on_image(binary_warped)
        lane_area = self.lane_det.render_lane_area()
        colored_lanes = self.lane_det.render_lanes()

        postprocessed_lane_area = self.postprocess_result(lane_area)
        postprocessed_lanes = self.postprocess_result(colored_lanes)

        image_with_area = cv2.addWeighted(undistorted_image, 1., postprocessed_lane_area, .3, 1)
        output_image = imgf.image_overlay(postprocessed_lanes, image_with_area, overlay_transparency=.1)
        return self.lane_det.display_metrics(output_image)

    def evaluate(self):
        undistorted_image, binary_warped = self.preprocess_image(self.input_image)
        self.lane_det.run_on_image(binary_warped)
        return self.lane_det.get_visualization()

    def preprocess_image(self, image):
        undistorted_image = self.undistorter.apply(image)

        hls_image = imgf.to_hls(undistorted_image)
        gray_image = imgf.to_grayscale(undistorted_image)

        s_threshold = imgf.color_threshold(hls_image, 2, (170, 255))
        gray_threshold = imgf.color_threshold((np.dstack((gray_image, gray_image, gray_image))), 0, (200, 255))

        combined_color_thresholds = imgf.combine_binary(s_threshold, gray_threshold)

        sobel = imgf.SobelGradientThresholder(undistorted_image, sobel_kernel=3)
        gradient_thresholds = sobel.abs_thresh(orient='x', thresh=(20, 100))

        thresholded_image = imgf.combine_binary(gradient_thresholds, combined_color_thresholds)

        binary_warped = self.warp(thresholded_image)
        return undistorted_image, binary_warped

    def postprocess_result(self, image):
        return self.unwarp(image)

    def warp(self, image):
        return self.transform_perspective(image, self.transform_M)

    def unwarp(self, image):
        return self.transform_perspective(image, self.transform_M_inv)

    def transform_perspective(self, image, M):
        image_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)


def get_transform_params(height, width):
    image_size = width, height

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
