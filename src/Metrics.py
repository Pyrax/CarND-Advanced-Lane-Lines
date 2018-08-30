import numpy as np


class Metrics:
    def __init__(self):
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 640 # meters per pixel in x dimension

    def measure_car_position(self, image, left_fit, right_fit, plot_y):
        # Define y-value where we want position of car
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(plot_y)

        left_lane_pos = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        right_lane_pos = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        lane_middle = np.abs(left_lane_pos - right_lane_pos)

        # Assume camera is centered in the middle of the car
        car_pos = image.shape[1] / 2

        center_dist = car_pos - lane_middle
        return center_dist * self.xm_per_pix

    def measure_lane_position(self, image, fit, plot_y):
        # Define y-value where we want position of car
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(plot_y)

        lane_pos = fit[0] * y_eval ** 2 + fit[1] * y_eval + fit[2]

        # Assume camera is centered in the middle of the car
        car_pos = image.shape[1] / 2

        lane_dist_from_center = np.abs(car_pos - lane_pos)
        return lane_dist_from_center * self.xm_per_pix

    def measure_curvature_pixels(self, left_fit, right_fit, plot_y):
        """
        Calculates the curvature of polynomial functions in pixels.
        """
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(plot_y)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        return left_curverad, right_curverad

    def measure_curvature_real(self, left_fit_cr, right_fit_cr, plot_y):
        """
        Calculates the curvature of polynomial functions in meters.
        """
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(plot_y)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
                         ) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
                          ) / np.absolute(2 * right_fit_cr[0])

        return left_curverad, right_curverad
