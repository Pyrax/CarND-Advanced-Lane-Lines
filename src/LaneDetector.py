import cv2
import numpy as np
from Metrics import Metrics
from LaneFindingStrategies import SlidingWindowStrategy, PolySearchStrategy


class Lane:
    def __init__(self):
        self.detected = False   # was the line detected in the last iteration?

        self.current_fit = [np.array([False])]  # polynomial coefficients for the most recent fit
        self.current_fit_x = None               # points for plotting polynomial

        self.radius_of_curvature = None     # radius of curvature of the line in meters
        self.line_base_pos = None           # distance in meters of vehicle center from the line

        self.all_x = None   # x values for detected line pixels
        self.all_y = None   # y values for detected line pixels

    def update(self, line_x, line_y, fit, fit_x, curvature, distance_from_center):
        self.detected = True

        self.all_x = line_x
        self.all_y = line_y

        self.current_fit = fit
        self.current_fit_x = fit_x

        self.radius_of_curvature = curvature
        self.line_base_pos = distance_from_center

    def get_pixels(self):
        return self.all_x, self.all_y

    def get_fit(self):
        return self.current_fit

    def get_fit_x(self):
        return self.current_fit_x


class LaneDetector:
    def __init__(self, n_windows=9, margin=100, min_pix=50, poly_margin=100):
        self.image = None
        self.last_used_strategy = None

        self.left_lane = Lane()
        self.right_lane = Lane()
        self.bad_frames = 0

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
        self.last_used_strategy = strategy
        left_x, left_y, right_x, right_y = strategy.find_lane(self.image,
                                                              self.left_lane.current_fit,
                                                              self.right_lane.current_fit)
        left_fit, right_fit, left_fit_x, right_fit_x = self.fit_poly(left_x, left_y, right_x, right_y)

        left_curverad, right_curverad = self.metrics.measure_curvature_real(left_fit, right_fit, self.plot_y)
        left_lane_pos = self.metrics.measure_lane_position(image, left_fit, self.plot_y)
        right_lane_pos = self.metrics.measure_lane_position(image, right_fit, self.plot_y)

        if self.passes_sanity_check(image,
                                    left_x, left_y, left_fit, left_fit_x, left_curverad, left_lane_pos,
                                    right_x, right_y, right_fit, right_fit_x, right_curverad, right_lane_pos):
            self.bad_frames = 0
            self.left_lane.update(left_x, left_y, left_fit, left_fit_x, left_curverad, left_lane_pos)
            self.right_lane.update(right_x, right_y, right_fit, right_fit_x, right_curverad, right_lane_pos)
        else:
            self.bad_frames += 1

    def determine_strategy(self):
        if self.bad_frames > 10:
            self.bad_frames = 0
            self.left_lane = Lane()
            self.right_lane = Lane()
            return self.sliding_window

        for lane in [self.left_lane, self.right_lane]:
            if not lane.detected:
                return self.sliding_window

        return self.poly_search

    def passes_sanity_check(self, image,
                            left_x, left_y, left_fit, left_fit_x, left_curverad, left_lane_pos,
                            right_x, right_y, right_fit, right_fit_x, right_curverad, right_lane_pos):
        """ Returns true if lane data seems to be realistic. """
        if not self.left_lane.detected or not self.right_lane.detected:
            return True

        # Check that they are separated by approximately the right distance horizontally
        max_d = 0.05
        if not (self.left_lane.line_base_pos - max_d <= left_lane_pos <= self.left_lane.line_base_pos + max_d):
            return False
        if not (self.right_lane.line_base_pos - max_d <= right_lane_pos <= self.right_lane.line_base_pos + max_d):
            return False

        return True

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
        left_curverad, right_curverad = self.left_lane.radius_of_curvature, self.right_lane.radius_of_curvature
        cv2.putText(display_image, f'Left curvature radius: {left_curverad:.4f}m', (40, 60),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_image, f'Right curvature radius: {right_curverad:.4f}m', (40, 100),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display car's position relative to the center
        center_dist = self.metrics.measure_car_position(self.image, self.left_lane.get_fit(), self.right_lane.get_fit(),
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

    def display_debug_info(self, display_image):
        font = cv2.FONT_HERSHEY_SIMPLEX

        def display_lane_debug_info(lane, name, x_pos, y_off):
            cv2.putText(display_image, name, (x_pos, y_off + 60), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_image, f'Curvature: {lane.radius_of_curvature:.2f}m', (x_pos, y_off + 100),
                        font, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_image, f'Distance from center: {lane.line_base_pos:.4f}m', (x_pos, y_off + 140),
                        font, 1, (255, 255, 0), 2, cv2.LINE_AA)

            return display_image

        for lane in [[self.left_lane, 'Left lane', 40, 0], [self.right_lane, 'Right lane', 40, 260]]:
            display_lane_debug_info(lane[0], lane[1], lane[2], lane[3])

        cv2.putText(display_image, f'Bad frames: {self.bad_frames}m', (40, 600), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        return display_image
