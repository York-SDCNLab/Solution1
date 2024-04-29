# python imports
import os
import sys
import time
# 3rd party imports
import cv2
# quanser imports
from qvl.qlabs import QuanserInteractiveLabs
# custom imports
from .utils import EventWrapper
from .qcar import VLFCar, EVLFControl, EVLFObserver
# single process
def start_car(debug: bool = False, throttle=0.1) -> None: 
    """
    Start the car with the vision line follow policy

    Parameters:
        debug (bool): Debug mode for the QCar.
        throttle (int): The throttle value for the QCar.

    Returns:
        None
    """
    try: 
        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        car: VLFCar = VLFCar()
        car.setup(throttle=throttle)
        
        while True: 
            car.execute()
            if debug: 
                cv2.waitKey(1)
    except KeyboardInterrupt:
        car.halt_car()
        os._exit(0) 

# multiprocessing for control and observer
def run_control_process(stop_event: EventWrapper, throttle: int = 0.1) -> None: 
    """
    Run the control process for the QCar.

    Parameters:
        stop_event (EventWrapper): The stop event for the QCar.
        throttle (int): The throttle value for the QCar.

    Returns:
        None
    """
    try: 
        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        # initialize the car
        control: EVLFControl = EVLFControl(stop_event)
        control.setup(throttle=throttle)
        print("Starting control process...")
        # get the data from queue 
        control.policy.start = time.time()
        while True: 
            control.execute()
            # cv2.waitKey(1)
    except KeyboardInterrupt:
        control.halt_car()
    except Exception as e:
        print(e)
    finally: 
        sys.exit(0)

def run_observer_process(stop_event: EventWrapper, activate_event) -> None: 
    """
    Run the observer process for the QCar.

    Parameters:
        stop_event (EventWrapper): The stop event for the QCar.
        activate_event (Event): The activate event for the QCar.

    Returns:
        None
    """
    try: 
        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        # initialize the car
        observer: EVLFObserver = EVLFObserver(
            events=stop_event, 
            file_path='plan_vision_linefollow/model_weights_final_1999.qcar'
        )
        # get the data from queue 
        activate_event.set()
        print("Starting observer process...")
        while True: 
            observer.execute()
            # cv2.waitKey(1)
    except KeyboardInterrupt:
        sys.exit(0)

