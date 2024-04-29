class NoContourException(Exception):
    """
    Exception raised when no contour is detected in an image processing operation.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message: str = "No contour found!") -> None:
        self.message = message
        super().__init__(self.message)


class NoImageException(Exception):
    """
    Exception raised for errors that occur when an image is expected but not provided.

    This can be used to signal issues such as a missing image file, an undefined image variable,
    or an invalid image path in image processing workflows.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class StopException(Exception): 
    """
    Exception raised when the car is required to stop.

    Attributes:
        stop_time (float): The time for which the car should stop.
    """
    
    def __init__(self, stop_time: float = 3.0, *args: object) -> None: 
        super().__init__(*args)
        self.stop_time: float = stop_time

