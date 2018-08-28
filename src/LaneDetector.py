import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np


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


class LaneFindingStrategy:
    def color_lane_pixels(self, target_image, left_x, left_y, right_x, right_y):
        out_image = np.copy(target_image)

        # Colors in the left and right lane regions
        out_image[left_y, left_x] = [255, 0, 0]
        out_image[right_y, right_x] = [0, 0, 255]

        return out_image

    def draw_lanes(self, target_image, left_fit_x, right_fit_x, plot_y):
        out_image = np.copy(target_image)

        # Plots the left and right polynomials on the lane lines
        left_line_coords = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        right_line_coords = np.array([np.transpose(np.vstack([right_fit_x, plot_y]))])
        cv2.polylines(out_image, np.int_([left_line_coords]), False, color=(255, 255, 0), thickness=2)
        cv2.polylines(out_image, np.int_([right_line_coords]), False, color=(255, 255, 0), thickness=2)

        return out_image


class SlidingWindowStrategy(LaneFindingStrategy):
    def __init__(self, n_windows, margin, min_pix):
        self.sliding_windows_left = None
        self.sliding_windows_right = None

        self.n_windows = n_windows
        self.margin = margin
        self.min_pix = min_pix

    def find_lane(self, image):
        # Take a histogram of the bottom half of the image
        image_height = image.shape[0]
        half_height = image_height // 2
        histogram = np.sum(image[half_height:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(image_height // self.n_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        left_x_current = left_x_base
        right_x_current = right_x_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        self.sliding_windows_left = []
        self.sliding_windows_right = []

        # Step through the windows one by one
        for window in range(self.n_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image_height - (window + 1) * window_height
            win_y_high = image_height - window * window_height
            win_x_left_low = left_x_current - self.margin
            win_x_left_high = left_x_current + self.margin
            win_x_right_low = right_x_current - self.margin
            win_x_right_high = right_x_current + self.margin

            self.sliding_windows_left.append([(win_x_left_low, win_y_low), (win_x_left_high, win_y_high)])
            self.sliding_windows_right.append([(win_x_right_low, win_y_low), (win_x_right_high, win_y_high)])

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_x_left_low) & (
                        nonzero_x < win_x_left_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_x_right_low) & (
                        nonzero_x < win_x_right_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.min_pix:
                left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > self.min_pix:
                right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        return left_x, left_y, right_x, right_y

    def visualize_lanes(self, image, left_x, left_y, right_x, right_y, left_fit_x, right_fit_x, plot_y):
        target_image = np.dstack((image, image, image)) * 255

        # Draw the sliding windows on the visualization image
        for idx in range(len(self.sliding_windows_left)):
            cv2.rectangle(target_image, self.sliding_windows_left[idx][0], self.sliding_windows_left[idx][1], (0, 255, 0), 2)
            cv2.rectangle(target_image, self.sliding_windows_right[idx][0], self.sliding_windows_right[idx][1], (0, 255, 0), 2)

        target_image = self.color_lane_pixels(target_image, left_x, left_y, right_x, right_y)
        return self.draw_lanes(target_image, left_fit_x, right_fit_x, plot_y)


class PolySearchStrategy(LaneFindingStrategy):
    def __init__(self, poly_margin):
        self.poly_margin = poly_margin

    def find_lane(self, image, left_fit, right_fit):
        # Grab activated pixels
        nonzero = image.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Set the area of search based on activated x-values within the +/- margin of our polynomial function
        left_lane_inds = ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] - self.poly_margin)) &
                          (nonzero_x < (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] + self.poly_margin)))
        right_lane_inds = ((nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] - self.poly_margin)) &
                           (nonzero_x < (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] + self.poly_margin)))

        # Get pixel coordinates
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        return left_x, left_y, right_x, right_y

    def visualize_lanes(self, image, left_x, left_y, right_x, right_y, left_fit_x, right_fit_x, plot_y):
        # Create an image to draw on and an image to show the selection window
        target_image = np.dstack((image, image, image)) * 255
        window_image = np.zeros_like(target_image)

        target_image = self.color_lane_pixels(target_image, left_x, left_y, right_x, right_y)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - self.poly_margin, plot_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + self.poly_margin, plot_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - self.poly_margin, plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + self.poly_margin, plot_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_image, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_image, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(target_image, 1, window_image, 0.3, 0)

        return self.draw_lanes(result, left_fit_x, right_fit_x, plot_y)


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