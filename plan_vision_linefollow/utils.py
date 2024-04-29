import math
from queue import Queue
from typing import Any

import cv2
import numpy as np
from hal.utilities.image_processing import ImageProcessing


class SignalFilter: 
    """
    A decaying sliding window filter for the traffic light signal detection.

    Attributes:
        buffer_size (int): Size of the buffer for the filter.
        buffer (Queue): Buffer for the filter.
        accumulator (float): Accumulator for the filter.
        decay_factor (float): Decay factor for the filter.
        threshold (float): Threshold for the filter.
    
    Methods:
        __call__(signal: float) -> str:
            Filters the signal and returns the traffic light status.
        clear() -> None:
            Clears the buffer and the accumulator for the filter.
    """

    def __init__(self, threshold: float, buffer_size: int) -> None:  
        """
        Initializes the SignalFilter class with the specified threshold and buffer size.

        Parameters:
            threshold (float): Threshold for the filter.
            buffer_size (int): Size of the buffer for the filter.   
        
        Returns:
            None
        """   
        self.buffer_size: int = buffer_size   
        self.buffer: Queue = Queue(self.buffer_size)
        self.accumulator: float = 0
        self.decay_factor: float = 0.95
        self.threshold: float = threshold

    def __call__(self, signal: float) -> str:
        """
        The call method for the SignalFilter class.

        Parameters:
            signal (float): Signal value for the filter.
        
        Returns:
            str: Traffic light status for the signal.
        """
        if self.buffer.full():
            self.accumulator -= round(self.buffer.get() * math.pow(self.decay_factor, self.buffer_size - 1), 10)
        self.buffer.put(signal)
        self.accumulator = round(self.accumulator * self.decay_factor + signal, 10)
        result: float = abs(self.accumulator) / self.buffer.qsize()
        # print(round(result, 10))
        if round(result, 10) > self.threshold:
            return "red light"
        return "green light"
    
    def clear(self) -> None:
        """
        Clears the buffer and the accumulator for the filter.

        Returns:
            None
        """
        self.buffer = Queue(self.buffer_size)
        self.accumulator = 0


class EventWrapper: 
    """
    The Event Wrapper class for the QCar.

    Attributes:
        event (Event): Event for the QCar.
        event_types (dict): Event types for the QCar.
    
    Methods:
        setup(event_names: list) -> None:
            Sets up the event types for the QCar.
        set(event_name: str) -> None:
            Sets the event for the QCar.
        clear(event_name: str) -> None:
            Clears the event for the QCar.
    """

    def __init__(self, manager) -> None:
        """
        Initializes the Event Wrapper class for the QCar.

        Parameters:
            manager (Manager): Manager for the QCar.

        Returns:
            None
        """
        self.event = manager.Event()
        self.event_types = manager.dict()

    def setup(self, event_names: list) -> None: 
        """
        Sets up the event types for the QCar.

        Parameters:
            event_names (list): List of event names for the QCar.

        Returns:
            None
        """
        for event_name in event_names: 
             self.event_types[event_name] = False

    def set(self, event_name: str) -> None:
        """
        Sets the event for the QCar.

        Parameters:
            event_name (str): Event name for the QCar.

        Returns:
            None
        """
        self.event.set()
        self.event_types[event_name] = True

    def clear(self, event_name: str) -> None:
        """
        Clears the event for the QCar.

        Parameters:
            event_name (str): Event name for the QCar.

        Returns:
            None
        """
        self.event_types[event_name] = False
        if True not in self.event_types.values(): 
            self.event.clear()


class StopEventWrapper:
    """
    The Stop Event Wrapper class for the QCar.

    Attributes:
        stop_event (Event): Stop event for the QCar.
        stop_time (Value): Stop time for the QCar.

    Methods:
        set(stop_time: float) -> None:
            Sets the stop time for the QCar.
        is_set() -> bool:
            Returns the stop event status for the QCar.
        clear() -> None:
            Clears the stop event for the QCar.
    """
    
    def __init__(self, manager) -> None:
        """
        Initializes the Stop Event Wrapper class for the QCar.

        Parameters:
            manager (Manager): Manager for the QCar.

        Returns:
            None
        """
        self.stop_event = manager.Event()
        # self.evet_type = manager.Value('i', 0)
        self.stop_time = manager.Value('d', 0.0)

    def set(self, stop_time: float) -> None:
        """
        Sets the stop time and the event for the QCar.

        Parameters:
            stop_time (float): Stop time for the QCar.

        Returns:
            None
        """
        self.stop_event.set()
        self.stop_time.value = stop_time

    def is_set(self) -> bool:
        """
        Checks the stop event status for the QCar.

        Returns:
            bool: Stop event status for the QCar.
        """
        return self.stop_event.is_set()
    
    def clear(self) -> None:
        """
        Clears the stop event for the QCar.
        """
        self.stop_event.clear()


class HorizontalDetector: 
    def __init__(self, threshold: float = 2000.0) -> None: 
        self.image: np.ndarray = None
        self.threshold: float = threshold

    def save_image(self, image: np.ndarray) -> None:
        self.image = image[290:, 100:320]
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def _find_hough_lines(self, image: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines: np.ndarray = cv2.HoughLines(edges, 1, np.pi/180, 50)
        # draw lines on the image
        self.found = False
        if lines is not None: 
            for line in lines: 
                rho: float = line[0][0]
                theta: float = line[0][1]
                angle: float = theta * 180 / np.pi
                if 90 <= angle <= 115: 
                    a: float = np.cos(theta)
                    b: float = np.sin(theta)
                    x0: float = a * rho
                    y0: float = b * rho
                    x1: int = int(x0 + 1000 * (-b))
                    y1: int = int(y0 + 1000 * (a))
                    x2: int = int(x0 - 1000 * (-b))
                    y2: int = int(y0 - 1000 * (a))
                    # draw the line on the image
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    self.found = True
        return image
    
    def _find_horizontal(self, image: np.ndarray) -> bool:
        blurred_image: np.ndarray = cv2.GaussianBlur(image, (9, 9), 0)
        blurred_image = ImageProcessing.image_filtering_open(blurred_image)
        thresh: np.ndarray = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # draw contours on the image
        if contours is not None and len(contours) != 0: 
            largest_contour: np.ndarray = max(contours, key=cv2.contourArea)
            if self.threshold <= cv2.contourArea(largest_contour) <= 1800: 
                cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)
                # cv2.drawContours(self.image, [largest_contour], -1, (0, 255, 0), 3)
                cv2.imshow("Image", self.image)
                print(cv2.contourArea(largest_contour))
                return True and self.found
        cv2.imshow("Image", self.image)
        return False
    
    def execute(self) -> Any:
        processed_image: np.ndarray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hough_image: np.ndarray = self._find_hough_lines(processed_image)
        # cv2.imshow("Horizontal Image", self.image)
        result = self._find_horizontal(hough_image)
        # self.found = False
        return result
