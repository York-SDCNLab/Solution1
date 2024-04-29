import cv2
import time
import numpy as np
from typing import Tuple

from quanser_pkgs.pal.utilities.math import Filter
from quanser_pkgs.hal.utilities.image_processing import ImageProcessing

class VisionLineFollowingPolicy:
    def __init__(self):
        #hough transform angles
        self.min_angle = 100
        self.max_angle = 180

        #image dimensions and parameters
        self.image_width = 820
        self.image_height = 410
        self.upper_crop = 270

        #PID gains and parameters
        self.kp = -0.8
        self.ki = 0.0
        self.kd = -0.06
        self.prev_cross_error = 0.0
        self.prev_integral_error = 0.0
        self.prev_derivative_error = 0.0

        #default line parameters if line offscreen
        self.slope_offset = -0.7718322998996283
        self.intersct_offset = 527.665286400669

        #filter
        self.steering_filter = Filter().low_pass_first_order_variable(25, 0.033)
        next(self.steering_filter)

        self.vel = 0.1
        self.prev_action = np.array([0.0, 0.0])
        self.prev_action_time = time.perf_counter()

    def __call__(self, obs):
        metrics = {}
        image = obs["image"][self.upper_crop:, :]

        '''houghline_image, edge_image = self.get_houghline_image(image)
        if houghline_image is None:
            return self.prev_action, metrics'''

        binary_image = self.get_lane_edge(image)
        if binary_image is None:
            return self.prev_action, metrics

        # Find slope and intercept of linear fit from the binary image
        slope, intercept = ImageProcessing.find_slope_intercept_from_binary(binary=binary_image)

        steering = self.PID(slope, intercept)
        velocity = self.vel * np.cos(2 * steering)

        action = np.array([velocity, steering])
        self.prev_action_time = time.perf_counter()
        self.prev_action = action

        return action, metrics

    def PID(
        self,
        slope: float,
        intercept: float
    ):
        # steering from slope and intercept
        if abs(slope) < 0.2 and abs(intercept) < 100:
            slope = self.slope_offset
            intercept = self.intersct_offset

        #PID Controller Design
        cross_err = (intercept / -slope) - (self.intersct_offset / -self.slope_offset)

        #error normalization and calculus
        cross_err = cross_err / self.image_width
        dt = time.perf_counter() - self.prev_action_time
        integral_error = self.prev_integral_error + (dt * cross_err)
        derivative_error = (cross_err - self.prev_cross_error) / dt

        raw_steer = (self.kp * cross_err) + (self.ki * integral_error) + (self.kd * derivative_error)
        if slope == 0.3419:
            steering = 0.0
        else:
            steering = self.steering_filter.send((np.clip(raw_steer, -0.5, 0.5), dt))

        #keep track of errors
        self.prev_cross_error = cross_err
        self.prev_integral_error = integral_error
        self.prev_derivative_error = derivative_error

        return steering

    def get_houghline_image(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            crop_image = image[self.upper_crop:, :]
            grey_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(grey_image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 95)

            for line in lines:
                rho, theta = line[0]
                angle = theta * (180 / np.pi)  

                if self.min_angle <= angle <= self.max_angle:  
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    cv2.line(grey_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.line(crop_image, (x1, y1), (x2, y2), (0, 0, 255), 2 )

            return grey_image, crop_image
        except Exception as e:
            return None, None

    def get_lane_edge(
        self,
        image: np.ndarray
    ):
        # Convert the hough_image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        blurred = ImageProcessing.image_filtering_open(blurred)

        # Apply thresholding to separate road and off-road areas
        thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        edge = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)

            # Create a mask for the road area
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [hull], (255, 255, 255))
            diff = cv2.Sobel(mask, cv2.CV_64F, 1, 1, ksize=15)
            edge = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)[1]

        return edge