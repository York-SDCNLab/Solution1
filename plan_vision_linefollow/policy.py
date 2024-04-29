# python imports 
import time 
# 3rd party module imports
import cv2 
# quanser imports
from pal.utilities.math import *
from hal.utilities.image_processing import ImageProcessing
# custom imports
from .exceptions import NoImageException, NoContourException


class VisualLineFollowing: 
    """
    Visual line following policy for the QCar.

    Attributes:
        slope_offset (float): Offset value for the slope of the line.
        intercept_offset (float): Offset value for the intercept of the line.
        integral_error (float): Integral error for the PID controller.
        cross_err (float): Cross error for the PID controller.
        pre_cross_err (float): Previous cross error for the PID controller.
        previous_derivative_term (float): Previous derivative term for the PID controller.
        start (float): Start time for the PID controller.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        steering (float): Steering value for the QCar.
        throttle (float): Throttle value for the QCar.
        steering_filter (Filter): Filter for the steering value.

    Methods:
        preprocess_image(image: np.ndarray) -> np.ndarray:
            Preprocesses the input image for line following.
        get_houghline_image(grey_image: np.ndarray) -> np.ndarray:
            Gets the Hough line image from the input image.
        find_conturs(image: np.ndarray) -> np.ndarray:
            Finds the contours in the input image.
        get_edge_image(image: np.ndarray, contours: np.ndarray) -> np.ndarray:
            Gets the edge image from the input image and contours.
        visual_steering_pid(input: tuple) -> float:
            Implements the PID controller for visual steering.
        execute_policy(origin_image: np.ndarray) -> None:
            Executes the visual line following policy on the input image.
    """

    def __init__(self, image_width: int = 820, image_height: int = 410, throttle: float = 0.08) -> None:
        """
        Initializes the VisualLineFollowing policy with the specified image width, image height, and throttle value.

        Parameters:
            image_width (int): Width of the input image.
            image_height (int): Height of the input image.
            throttle (float): Throttle value for the QCar.
        """
        self.slope_offset: float = -0.7718322998996283
        self.intercept_offset: float = 450 # 450.1033252934936 # 527.665286400669
        self.integral_error: float = 0.0
        self.cross_err: float = 0.0
        self.pre_cross_err: float = 0.0
        self.pre_steering: float = 0.0
        self.previous_derivative_term: float = 0.0
        self.image_width: int = image_width
        self.image_height: int = image_height
        self.steering: float = 0.0
        self.start: float = 0.0
        self.throttle: float = throttle
        self.steering_filter = Filter().low_pass_first_order_variable(90, 0.01)
        next(self.steering_filter)

    def setup(self, k_p: float = -1.02, k_i: float = -0.000, k_d:float = -0.15) -> None: #self, k_p: float = -1.2, k_i: float = -0.000, k_d:float = -0.15
        """
        Sets up the PID controller with the specified gains.

        Parameters:
            k_p (float): Proportional gain for the PID controller.
            k_i (float): Integral gain for the PID controller.
            k_d (float): Derivative gain for the PID controller.

        Returns:
            None
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.start = time.time()
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input image for line following.

        Parameters:
            image (np.ndarray): Input image for line following.

        Returns:
            np.ndarray: Preprocessed image for line following.
        """
        # check if the image is None
        if image is None: 
            raise NoImageException()
        # crop the image
        self.image: np.ndarray = image.copy() 
        cropped_image: np.ndarray = image[270:, 100:]
        # convert the image to grayscale
        gray_image: np.ndarray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        return gray_image

    def get_houghline_image(self, grey_image: np.ndarray) -> np.ndarray:
        """
        Gets the Hough line image from the input image.

        Parameters:
            grey_image (np.ndarray): Greyscale image for line following.

        Returns:
            np.ndarray: Hough line image for line following.
        """
        if grey_image is None: 
            raise NoImageException()
        # thresholds of the line angles
        min_angle: float = 100
        max_angle: float = 160
        edges: np.ndarray = cv2.Canny(grey_image, 50, 150, apertureSize=3)
        lines: np.ndarray = cv2.HoughLines(edges, 1, np.pi/180, 85)
        if lines is None: 
            # cv2.imshow("HoughLine", grey_image)
            return grey_image 
        # calculate the line parameters
        for line in lines: 
            rho: float = line[0][0]
            theta: float = line[0][1]
            angle: float = theta * 180 / np.pi
            if min_angle <= angle <= max_angle: 
                a: float = np.cos(theta)
                b: float = np.sin(theta)
                x0: float = a * rho
                y0: float = b * rho
                x1: int = int(x0 + 1000 * (-b))
                y1: int = int(y0 + 1000 * (a))
                x2: int = int(x0 - 1000 * (-b))
                y2: int = int(y0 - 1000 * (a))
                # draw the line on the image
                cv2.line(grey_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imshow("HoughLine", self.image)
        return grey_image

    def find_conturs(self, image: np.ndarray) -> np.ndarray:
        """
        Finds the contours in the input image.

        Parameters:
            image (np.ndarray): Input image for line following.

        Returns:
            np.ndarray: Contours found in the input image.
        """
        if image is None: 
            raise NoImageException()
        # gaussian blur the image
        blurred_image: np.ndarray = cv2.GaussianBlur(image, (9, 9), 0)
        blurred_image = ImageProcessing.image_filtering_open(blurred_image)
        # threshold the image
        thresh: np.ndarray = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)[1]
        # find the contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours
    
    def get_edge_image(self, image: np.ndarray, contours: np.ndarray) -> np.ndarray:
        """
        Gets the edge image from the input image and contours.

        Prameters:
            image (np.ndarray): Input image for line following.
            contours (np.ndarray): Contours found in the input image.

        Returns:
            np.ndarray: Edge image for line following.
        """
        if image is None or contours is None or len(contours) == 0: 
            raise NoContourException()
        # find the largest contour
        largest_contour: np.ndarray = max(contours, key=cv2.contourArea)
        # cv2.drawContours(self.image, [largest_contour], -1, (0, 255, 0), 3)
        # draw the largest contour on the image
        hull: np.ndarray = cv2.convexHull(largest_contour)
        # draw the hull on the image
        mask: np.ndarray = np.zeros_like(image)
        cv2.fillPoly(mask, [hull], (255, 255, 255))
        # cv2.imshow("Mask", mask)
        # Calculate the difference between adjacent pixels
        diff: np.ndarray = cv2.Sobel(mask, cv2.CV_64F, 1, 1, ksize=15)
        edge: np.ndarray = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("Edge", edge)
        # cv2.imshow("Largest Contour", self.image)
        return edge
    
    def visual_steering_pid(self, input: tuple) -> float:
        """
        PID controller for visual steering.

        Parameters:
            input (tuple): Tuple containing the slope and intercept of the line.

        Returns:    
            float: Steering value for the QCar.
        """
        
        slope: float = input[0]
        intercept: float = input[1]
        # print(f"slope: {slope}, intercept: {intercept}")
        if slope == 0.3419:
            return 0.0
        # sterring from slope and intercept
        if abs(slope) < 0.2 and abs(intercept) < 100:
            slope = self.slope_offset
            intercept = self.intercept_offset
            print("Parking lot!\n")
        # PID control design 
        self.cross_err: float = (intercept/-slope) - (self.intercept_offset / -self.slope_offset)
        self.cross_err = self.cross_err / self.image_width
        dt = time.time() - self.start
        control_rate = 1 / dt
        self.steering_filter = Filter().low_pass_first_order_variable(control_rate-5, dt, self.pre_steering)
        next(self.steering_filter)
        
        self.start = time.time()
        self.integral_error += dt * self.cross_err
        derivetive_error: float = (self.cross_err-self.pre_cross_err) / dt
        if dt > 0.1: 
            dt = 0.033
        # steering filter
        raw_steering: float = self.k_p * self.cross_err + self.k_i * self.integral_error + self.k_d * derivetive_error 
        steering: float = self.steering_filter.send((np.clip(raw_steering, -0.5, 0.5), dt))
        self.pre_steering: float = steering
        self.steering_filter.close()
        # save the last cross error
        self.pre_cross_err = self.cross_err
        self.previous_derivative_term = derivetive_error
        # print(f"steering: {steering}")
        return steering

    def execute_policy(self, origin_image: np.ndarray) -> None:
        """
        The visual line following policy execution function for the QCar.

        Parameters:
            origin_image (np.ndarray): Input image from the QCar.

        Returns:
            None
        """
        # preprocess the image
        preprocessed_image: np.ndarray = self.preprocess_image(origin_image)
        # get the hough line image
        houghline_image: np.ndarray = self.get_houghline_image(preprocessed_image)
        # find the contours
        contours: np.ndarray = self.find_conturs(houghline_image)
        # get the edge image
        binary_image: np.ndarray = self.get_edge_image(houghline_image, contours)
        # calculate the angle
        res: tuple = ImageProcessing.find_slope_intercept_from_binary(binary=binary_image)
        # pid control steering
        self.steering = self.visual_steering_pid(res)
        