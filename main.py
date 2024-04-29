# python imports
import sys
import time
import random
from multiprocessing import Process, Event, Manager
# modeule imports
from path_planning.acc_roadmap import ACCRoadMap 
from scripts.traffic_light import run_traffic_light
# from plan_vision_linefollow.script import writer_tasks, car_tasks
from plan_vision_linefollow import start_car, run_control_process, run_observer_process
from plan_vision_linefollow.utils import EventWrapper
from scripts.Setup_Competition import setup
# from collection_demo import *

def spawn_on_node(node_id=24) -> None: 
    """
    Spawn the car on the specified node (0 - 24). 

    Parameters:
        node_id (int): The node id to spawn the car on.
    """
    # init the roadmap
    roadmap: ACCRoadMap = ACCRoadMap() 
    x_pos, y_pose, angle = roadmap.nodes[node_id].pose
    setup(initialPosition=[x_pos, y_pose, 0], initialOrientation=[0, 0, angle]) 

def start_multiprocess_car(node_id: int = 24, throttle: float = 0.1) -> None:
    """
    Start the car with the vision line follow policy and the detection process.

    Parameters:
        node_id (int): The node id to spawn the car on.
        throttle (float): The throttle value for the car.
    """
    spawn_on_node(node_id=node_id) # spawn the car on the specific node
    manager = Manager()
    events = ["stop_sign", "horizontal_line", "red_light", "unknown_error"]
    events_warpper: EventWrapper = EventWrapper(manager)
    events_warpper.setup(events)
    activate_event = Event()
    # initialize processes
    control_process: Process = Process(
        target=run_control_process, 
        args=(events_warpper, throttle)
    ) # change the second argument of control_process to change the throttle
    observer_process: Process = Process(
        target=run_observer_process, 
        args=(events_warpper, activate_event)
    )
    traffic_light_process = Process(
        target=run_traffic_light, 
        args=('auto', 1.5, None))
    activate_event.clear()
    traffic_light_process.start()
    time.sleep(4) # wait for the traffic light process to start
    observer_process.start()
    while not activate_event.is_set():
        time.sleep(0.01)
        continue
    activate_event.clear()
    control_process.start()
    try: 
        # run_traffic_light(mode='manual', sleep_time=1.5, process=control_process)
        while True: 
            time.sleep(100)
    except KeyboardInterrupt: 
        control_process.terminate()
        observer_process.terminate()
        # traffic_light_process.terminate()
    except Exception as e:
        print(e)
    finally:
        sys.exit(0)

if __name__ == "__main__":
    # Uncomment the spawn on node and traffic light process if you want
    start_multiprocess_car(node_id=24, throttle=0.13) # Recommended throttle value is 0.13