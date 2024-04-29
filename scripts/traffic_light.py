import time
import math
import random

from threading import Thread
from qvl.qlabs import QuanserInteractiveLabs
from qvl.traffic_light import QLabsTrafficLight

from scripts.Setup_Competition import setup

def run_traffic_light(mode = 'auto', sleep_time: int = 5, process= None) -> None:
    sleep_time = 5
    # creates a server connection with Quanser Interactive Labs and manages the communications
    qlabs = QuanserInteractiveLabs()

    print("Connecting to QLabs...")
    # trying to connect to QLabs and open the instance we have created - program will end if this fails
    try:
        qlabs.open("localhost")
    except:
        print("Unable to connect to QLabs")

    # traffic light
        
    x_offset = 0.13
    y_offset = 1.67

    TrafficLight0 = QLabsTrafficLight(qlabs)
    TrafficLight1 = QLabsTrafficLight(qlabs)
    TrafficLight0.spawn_degrees([2.3 + x_offset, y_offset, 0], [0, 0, 0], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
    TrafficLight1.spawn_degrees([-2.3 + x_offset, -1 + y_offset, 0], [0, 0, 180], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
    if mode != 'manual':
        TrafficLight0.set_state(QLabsTrafficLight.STATE_RED)
        TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)
    else:
        TrafficLight0.set_state(QLabsTrafficLight.STATE_RED)
        TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)

    if process is not None:
        process.start()

    i = 0
    while (True):
        # print (i)
        if mode == 'manual': 
            read = input("Press enter to change the traffic light: ")
            if read == "":
                i = i + 1
            if i % 2 == 0:
                print("Changing to RED")
                TrafficLight0.set_state(QLabsTrafficLight.STATE_RED)
                TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)
            else: 
                print("Changing to GREEN")
                TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
                TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)
        else: 
            i=i+1
            if i % 2 == 0:
                if mode == 'red_only':
                    TrafficLight0.set_state(QLabsTrafficLight.STATE_RED)
                    TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)
                    # print("0: RED", "1: RED")
                    time.sleep(sleep_time)
                    continue
                elif mode == 'green_only':
                    TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
                    TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)
                    # print("0: Green", "1: Green")
                    time.sleep(sleep_time)
                    continue
                TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
                TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)
                # print("0: Green", "1: RED")
            else:
                if mode == 'red_only':
                    TrafficLight0.set_state(QLabsTrafficLight.STATE_RED)
                    TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)
                    # print("0: RED", "1: RED")
                    time.sleep(sleep_time)
                    continue
                elif mode == 'red_only':
                    TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
                    TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)
                    # print("0: Green", "1: Green")
                    time.sleep(sleep_time)
                    continue
                TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)
                TrafficLight0.set_state(QLabsTrafficLight.STATE_RED)
                # print("0: RED", "1: Green")
        if mode != 'manual': 
            time.sleep(sleep_time)

    qlabs.close()
    print("Done!")

def manual_traffic_light(process, initial_state: str = "") -> None: 
    # creates a server connection with Quanser Interactive Labs and manages the communications
    qlabs = QuanserInteractiveLabs()

    print("Connecting to QLabs...")
    # trying to connect to QLabs and open the instance we have created - program will end if this fails
    try:
        qlabs.open("localhost")
    except:
        print("Unable to connect to QLabs")

    # traffic light
        
    x_offset = 0.13
    y_offset = 1.67

    TrafficLight0 = QLabsTrafficLight(qlabs)
    TrafficLight0.spawn_degrees([2.3 + x_offset, y_offset, 0], [0, 0, 0], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
    TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
    TrafficLight1 = QLabsTrafficLight(qlabs)
    TrafficLight1.spawn_degrees([-2.3 + x_offset, -1 + y_offset, 0], [0, 0, 180], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
    TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)
    # initial state
    if initial_state == "red":
        red_flag = True
    elif initial_state == "green":
        red_flag = False
    else:
        red_flag = (random.randint(0, 1) == 0)
    print(f"Initial traffic light state: {'red' if red_flag else 'green'}")
    process.start()
    read = None
    while True: 
        if read == "": 
            red_flag = not red_flag
            print("Traffic light changed to red" if red_flag else "Traffic light changed to green")

        if red_flag:
            TrafficLight0.set_state(QLabsTrafficLight.STATE_RED)
            TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)
        else:
            TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
            TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)
        
        read = input("Press enter to change the traffic light: ")
    
    qlabs.close()
    print("Done!")

if __name__ == "__main__":
    # temp function to test traffic light
    setup(initialPosition=[1.025,-0.745,0], initialOrientation=[0,0,math.pi])
    manual_traffic_light()