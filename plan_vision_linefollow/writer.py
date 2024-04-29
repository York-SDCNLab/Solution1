# python imports 
import os
import csv
# 3rd party imports
import cv2
import numpy as np


class DataWriter: 
    """
    Data writer class for recording training data.
    
    Attributes:
        data_path (str): Path to the training data folder.
        csv_filepath (str): Path to the training data CSV file.
        counter (int): Counter for the training data.
    
    Methods:
        setup() -> None:
            Sets up the training data folder.
        record_data(image_path: str, speed: float, steering: float) -> None:
            Records the training data to the CSV file.
        execute(image: np.ndarray, speed: float, steering: float) -> None:
            Executes the data recording operation.
    """

    def __init__(self, folder_name: str = 'training_data', csv_name: str = 'training_data') -> None:
        """
        Initializes the DataWriter class with the specified folder name and CSV name.

        Parameters:
            folder_name (str): Name of the training data folder.
            csv_name (str): Name of the training data CSV file.
        """
        current_path: str = os.getcwd()
        self.data_path: str = os.path.join(current_path, folder_name)
        self.csv_filepath: str = os.path.join(self.data_path, csv_name)
        self.counter: int = 0

    def setup(self) -> None: 
        """
        Sets up the training data folder.

        Returns:
            None
        """
        if not os.path.exists(self.data_path): 
            os.makedirs(self.data_path)

    def record_data(self, image_path: str, speed: float, steering: float = None) -> None: 
        """
        Records the training data to the CSV file.

        Parameters:
            image_path (str): Path to the training image.
            speed (float): Speed value for the QCar.
            steering (float): Steering value for the QCar.

        Returns:
            None
        """
        with open(self.csv_filepath, 'a', newline='') as csvfile: 
            writer = csv.writer(csvfile) 
            writer.writerow([image_path, speed, steering])

    def execute(self, image: np.ndarray, speed: float, steering: float) -> None: 
        """
        The data recording operation for the Device.

        Parameters:
            image (np.ndarray): Input image for the QCar.
            speed (float): Speed value for the QCar.
            steering (float): Steering value for the QCar.

        Returns:    
            None
        """
        image_path: str = os.path.join(self.data_path, 'image_{}.jpg'.format(self.counter) ) 
        self.record_data(image_path, speed, steering)
        cv2.imwrite(image_path, image)
        self.counter += 1