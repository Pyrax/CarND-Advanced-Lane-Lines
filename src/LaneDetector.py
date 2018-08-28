import cv2
import numpy as np
from LaneFindingStrategies import SlidingWindowStrategy, PolySearchStrategy


class Lane:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_x_fitted = []
        # average x values of the fitted line over the last n iterations
        self.best_x = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.all_x = None
        # y values for detected line pixels
        self.all_y = None


class LaneDetector:
    def __init__(self, n_windows=9, margin=100, min_pix=50, poly_margin=100):
        self.left_fit_x, self.right_fit_x, self.plot_y = None, None, None
        self.left_x, self.left_y, self.right_x, self.right_y = None, None, None, None
        self.left_fit, self.right_fit = None, None
        self.image = None
        self.last_used_strategy = None

        # TODO: use lane class to store the properties above
        self.left_lane = Lane()
        self.right_lane = Lane()

        # Adjustable parameters for sliding windows:
        self.sliding_window = SlidingWindowStrategy(n_windows, margin, min_pix)

        # Parameters for searching around polynomial function:
        self.poly_search = PolySearchStrategy(poly_margin)

    def run_on_image(self, image):
        self.image = image

        strategy = self.sliding_window if self.last_used_strategy is None else self.poly_search
        self.last_used_strategy = strategy

        left_x, left_y, right_x, right_y = self.find_lane_pixels(strategy)
        self.left_x, self.left_y, self.right_x, self.right_y = left_x, left_y, right_x, right_y

        left_fit, right_fit, left_fit_x, right_fit_x, plot_y = self.fit_poly(image, left_x, left_y, right_x, right_y)
        self.left_fit, self.right_fit = left_fit, right_fit
        self.left_fit_x, self.right_fit_x, self.plot_y = left_fit_x, right_fit_x, plot_y

        return left_fit_x, right_fit_x, plot_y

    def find_lane_pixels(self, strategy):
        if strategy is self.poly_search:
            return strategy.find_lane(self.image, self.left_fit, self.right_fit)
        return strategy.find_lane(self.image)

    def get_visualization(self):
        return self.last_used_strategy.visualize_lanes(self.image,
                                                       self.left_x, self.left_y, self.right_x, self.right_y,
                                                       self.left_fit_x, self.right_fit_x, self.plot_y)

    def get_lane_area(self):
        # Create an image to draw the lines on
        zeroed_image = np.zeros_like(self.image).astype(np.uint8)
        marked_image = np.dstack((zeroed_image, zeroed_image, zeroed_image))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fit_x, self.plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fit_x, self.plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(marked_image, np.int_([pts]), (0, 255, 0))
        return marked_image

    def get_lanes(self):
        zeros = np.zeros_like(self.image).astype(np.uint8)
        lanes_image = np.dstack((zeros, zeros, zeros))
        return self.last_used_strategy.color_lane_pixels(lanes_image,
                                                         self.left_x, self.left_y, self.right_x, self.right_y)

    def fit_poly(self, image, left_x, left_y, right_x, right_y):
        # Fit a second order polynomial to each lane using `np.polyfit`
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        # Generate x and y values for plotting
        plot_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        try:
            left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
            right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fit_x = 1 * plot_y ** 2 + 1 * plot_y
            right_fit_x = 1 * plot_y ** 2 + 1 * plot_y

        return left_fit, right_fit, left_fit_x, right_fit_x, plot_y


# class CurvatureCalculator: def measure_curvature(right_fit, left_fit): return left_curverad, right_curverad