import cv2
import numpy as np
from Metrics import Metrics
from LaneFindingStrategies import SlidingWindowStrategy, PolySearchStrategy


class Lane:
    def __init__(self):
        self.detected = False   # was the line detected in the last iteration?

        self.best_x = None                              # average x values of the fitted line over the last n iterations
        self.recent_x_fitted = []                       # x values of the last n fits of the line
        self.best_fit = None                            # polynomial coefficients averaged over the last n iterations
        self.current_fit = [np.array([False])]          # polynomial coefficients for the most recent fit
        self.current_fit_x = None                       # points for plotting polynomial
        self.diffs = np.array([0,0,0], dtype='float')   # difference in fit coefficients between last and new fits

        self.radius_of_curvature = None     # radius of curvature of the line in some units
        self.line_base_pos = None           # distance in meters of vehicle center from the line

        self.all_x = None   # x values for detected line pixels
        self.all_y = None   # y values for detected line pixels

    def update(self, line_x, line_y, fit, fit_x, plot_y):
        self.all_x = line_x
        self.all_y = line_y

        if fit is None:
            self.detected = False
            self.current_fit = [np.array([False])]
        else:
            self.current_fit = fit

        self.current_fit_x = fit_x

    def get_pixels(self):
        return self.all_x, self.all_y

    def get_coefficients(self):
        return self.current_fit

    def get_fit_x(self):
        return self.current_fit_x


class LaneDetector:
    def __init__(self, n_windows=9, margin=100, min_pix=50, poly_margin=100):
        self.image = None
        self.last_used_strategy = None

        self.left_lane = Lane()
        self.right_lane = Lane()

        self.metrics = Metrics()

        # Adjustable parameters for sliding windows:
        self.sliding_window = SlidingWindowStrategy(n_windows, margin, min_pix)
        # Parameters for searching around polynomial function:
        self.poly_search = PolySearchStrategy(poly_margin)

    def run_on_image(self, image):
        self.image = image

        # Generate y values for plotting through image height
        self.plot_y = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])

        strategy = self.determine_strategy()
        left_x, left_y, right_x, right_y = strategy.find_lane(self.image,
                                                              self.left_lane.current_fit,
                                                              self.right_lane.current_fit)
        left_fit, right_fit, left_fit_x, right_fit_x = self.fit_poly(left_x, left_y, right_x, right_y)

        self.left_lane.update(left_x, left_y, left_fit, left_fit_x, self.plot_y)
        self.right_lane.update(right_x, right_y, right_fit, right_fit_x, self.plot_y)

    def determine_strategy(self):
        default_strategy = self.poly_search

        def find_other_strategy_by_lane_data(lanes):
            for lane in lanes:
                if not lane.detected:
                    return self.sliding_window

        better_strategy = find_other_strategy_by_lane_data([self.left_lane, self.right_lane])

        strategy = default_strategy if not better_strategy else better_strategy
        self.last_used_strategy = strategy
        return strategy

    def get_visualization(self):
        return self.last_used_strategy.visualize_lanes(self.image,
                                                       *self.left_lane.get_pixels(),
                                                       *self.right_lane.get_pixels(),
                                                       self.left_lane.get_fit_x(),
                                                       self.right_lane.get_fit_x(),
                                                       self.plot_y)

    def render_lane_area(self):
        # Create an image to draw the lines on
        zeroed_image = np.zeros_like(self.image).astype(np.uint8)
        marked_image = np.dstack((zeroed_image, zeroed_image, zeroed_image))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_lane.get_fit_x(), self.plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_lane.get_fit_x(), self.plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(marked_image, np.int_([pts]), (0, 255, 0))
        return marked_image

    def render_lanes(self):
        zeros = np.zeros_like(self.image).astype(np.uint8)
        lanes_image = np.dstack((zeros, zeros, zeros))
        return self.last_used_strategy.color_lane_pixels(lanes_image,
                                                         *self.left_lane.get_pixels(),
                                                         *self.right_lane.get_pixels())

    def display_metrics(self, display_image):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Display the curvature
        left_curverad, right_curverad = self.metrics.measure_curvature_real(self.left_lane.get_coefficients(),
                                                                            self.right_lane.get_coefficients(),
                                                                            self.plot_y)
        cv2.putText(display_image, f'Left curvature radius: {left_curverad:.4f}m', (40, 60),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_image, f'Right curvature radius: {right_curverad:.4f}m', (40, 100),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display car's position relative to the center
        center_dist = self.metrics.measure_car_position(self.image,
                                                        self.left_lane.get_coefficients(),
                                                        self.right_lane.get_coefficients(),
                                                        self.plot_y)
        cv2.putText(display_image, f'Position to center: {center_dist:.4f}m', (40, 140), font, 1, (255, 255, 255),
                    2, cv2.LINE_AA)
        return display_image

    def fit_poly(self, left_x, left_y, right_x, right_y):
        # Fit a second order polynomial to each lane using `np.polyfit`
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        # Generate x values for plotting
        try:
            left_fit_x = left_fit[0] * self.plot_y ** 2 + left_fit[1] * self.plot_y + left_fit[2]
            right_fit_x = right_fit[0] * self.plot_y ** 2 + right_fit[1] * self.plot_y + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fit_x = 1 * self.plot_y ** 2 + 1 * self.plot_y
            right_fit_x = 1 * self.plot_y ** 2 + 1 * self.plot_y

        return left_fit, right_fit, left_fit_x, right_fit_x
