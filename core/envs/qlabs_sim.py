# standard imports
import os
import cv2
import time
import struct
import signal
import numpy as np
from threading import Thread
from typing import Dict, List, Tuple, Callable

# quanser imports
from quanser_pkgs.qvl.qcar import QLabsQCar
from quanser_pkgs.qvl.actor import QLabsActor
from quanser_pkgs.qvl.walls import QLabsWalls
from quanser_pkgs.qvl.system import QLabsSystem
from quanser_pkgs.qvl.flooring import QLabsFlooring
from quanser_pkgs.qvl.real_time import QLabsRealTime
from quanser_pkgs.qvl.stop_sign import QLabsStopSign
from quanser_pkgs.qvl.real_time import QLabsRealTime
from quanser_pkgs.qvl.crosswalk import QLabsCrosswalk
from quanser_pkgs.qvl.basic_shape import QLabsBasicShape
from quanser_pkgs.qvl.traffic_light import QLabsTrafficLight
from quanser_pkgs.qvl.qlabs import QuanserInteractiveLabs, CommModularContainer

# pal resources
from quanser_pkgs.pal.products.qcar import QCar
import quanser_pkgs.pal.resources.rtmodels as rtmodels

# acc imports
from scripts.Setup_Competition import setup
from path_planning.acc_roadmap import ACCRoadMap

# core imports
from core.sensors.rgbd_camera import VirtualRGBDCamera
from core.sensors.csi_camera import VirtualCSICamera


class FSM:
    def __init__(self, transitions: Dict[int, List[Tuple[int, float, Callable]]] = {}, initial_state=0):
        self.state = initial_state
        self.duration = 0

        self.transitions = transitions

    def step(self):
        self.duration += 1
        for state, penalty, condition in self.transitions[self.state]:
            if condition():
                # print("{} => {} with {}".format(self.state, state, penalty))
                self.state = state
                score = penalty * 30 / self.duration
                self.duration = 0

                return score

        return 0.0  # no penalty for keeping state


class QLabsSim:
    def __init__(self, x_offset: float = 0.13, y_offset: float = 1.67, dt=0.1, action_size=2, privileged=False):
        self.x_offset = x_offset
        self.y_offset = y_offset
        initial_position = [-1.335 + x_offset, -2.5 + y_offset, 0.005]
        initial_orientation = [0, 0, -np.pi / 4]
        self.dt = dt
        self.action_size = action_size
        self.prev_step_time = time.perf_counter()

        # initialize car
        self.car_id = 0
        self.car_qvl = None  # setup(initial_position, initial_orientation)
        self.car = None
        self.qlabs = QuanserInteractiveLabs()
        self.qlabs.open("localhost")

        # qlabs interface classes
        self.system_qvl = QLabsSystem(self.qlabs)
        self.floor_qvl = QLabsFlooring(self.qlabs)
        self.walls_qvl = QLabsWalls(self.qlabs)
        # self.car_qvl = QLabsQCar(self.qlabs)
        self.stop_sign_qvl = QLabsStopSign(self.qlabs)
        self.crosswalk_qvl = QLabsCrosswalk(self.qlabs)
        self.shape_qvl = QLabsBasicShape(self.qlabs)

        self.traffic_light1_qvl = QLabsTrafficLight(self.qlabs)
        self.traffic_light2_qvl = QLabsTrafficLight(self.qlabs)
        self.traffic_light_terminate = False
        self.traffic_light_thread = Thread(target=self.traffic_light_control)
        self.traffic_light_states = {
            0: "GREEN",
            1: "RED"
        }
        # self.traffic_light_thread.start()

        # car info
        self.privileged = privileged
        self.state = np.zeros((6,))
        self.in_stopsign_region = False
        self.in_trafficstop_region = False
        self.FSM = FSM({
            0: [
                (1, 0.0, lambda: self.get_in_stopsign_region()),
                (2, 0.0, lambda: self.get_in_trafficstop_region()),
            ],
            1: [
                (0, 10.0, lambda: self.state[3] > 0.05 and not self.in_stopsign_region),
                (3, 0.0, lambda: self.get_state_duration() >= 30)
            ],
            2: [
                (0, 10.0, lambda: self.state[3] > 0.05 and not self.in_trafficstop_region),
                (3, 0.0, lambda: not self.get_in_trafficstop_region())
            ],
            3: [(0, 0.0, lambda: not self.in_stopsign_region and not self.get_in_trafficstop_region())]
        })
        self.state_machine = 0  # 0 => Normal, 1 => In Stop Region (waiting), 2 => Stop Region Condition Met

        # environment info
        self.stop_signs = []
        self.traffic_lights = []
        self.crosswalks = []

        # sim info
        self.ep_steps = 0
        self.max_ep_steps = int(30 / self.dt)  # each episode should be 60 seconds

        # quanser objects
        self.basicshape = QLabsBasicShape(self.qlabs)

        # setup roadmap
        self.roadmap = ACCRoadMap()

        # valid closed sequences
        self.sequences = {
            0: np.array([10, 2, 4, 14, 20, 22, 10], dtype=np.int32),
            # 1: np.array([10, 2, 4, 6, 8, 10], dtype=np.int32),
            # 2: np.array([2, 4, 14, 16, 18, 11, 12, 0, 2], dtype=np.int32),
            # 3: np.array([9, 7, 14, 20, 22, 9], dtype=np.int32)
        }
        self.waypoint_sequence = None
        self.next_waypoints = None
        self.goal = None
        self.goal_thresh = 0.1  # distance that the vehicle needs to be from goal to end episode
        self.max_lookahead_indices = 200

        # start state thread
        # state_thread = Thread(target=self.get_state, args=[160, 0])
        # state_thread.setDaemon(True)
        # state_thread.start()
        self.sent = False

        # initialize the environment
        # self.initialize_environment()

    def initialize_environment(self):
        # Delete any previous QCar instances and stop any running spawn models
        self.qlabs.destroy_all_spawned_actors()
        QLabsRealTime().terminate_all_real_time_models()

        # Set the Workspace Title
        x = self.system_qvl.set_title_string('ACC Self Driving Car Competition', waitForConfirmation=True)

        ### Flooring
        self.floor_qvl.spawn_degrees([self.x_offset, self.y_offset, 0.001], rotation=[0, 0, -90])

        ### Walls
        self.walls_qvl.set_enable_dynamics(False)

        for y in range(5):
            self.walls_qvl.spawn_degrees(location=[-2.4 + self.x_offset, (-y * 1.0) + 2.55 + self.y_offset, 0.001],
                                         rotation=[0, 0, 0])

        for x in range(5):
            self.walls_qvl.spawn_degrees(location=[-1.9 + x + self.x_offset, 3.05 + self.y_offset, 0.001],
                                         rotation=[0, 0, 90])

        for y in range(6):
            self.walls_qvl.spawn_degrees(location=[2.4 + self.x_offset, (-y * 1.0) + 2.55 + self.y_offset, 0.001],
                                         rotation=[0, 0, 0])

        for x in range(5):
            self.walls_qvl.spawn_degrees(location=[-1.9 + x + self.x_offset, -3.05 + self.y_offset, 0.001],
                                         rotation=[0, 0, 90])

        self.walls_qvl.spawn_degrees(location=[-2.03 + self.x_offset, -2.275 + self.y_offset, 0.001],
                                     rotation=[0, 0, 48])
        self.walls_qvl.spawn_degrees(location=[-1.575 + self.x_offset, -2.7 + self.y_offset, 0.001],
                                     rotation=[0, 0, 48])

        # stop signs
        # myStopSign = QLabsStopSign(self.qlabs)
        self.stop_signs = []
        self.stop_signs.append(
            self.stop_sign_qvl.spawn_degrees([2.25 + self.x_offset, 1.5 + self.y_offset, 0.05], [0, 0, -90],
                                             [0.1, 0.1, 0.1], False)[1])
        self.stop_signs.append(
            self.stop_sign_qvl.spawn_degrees([-1.3 + self.x_offset, 2.9 + self.y_offset, 0.05], [0, 0, -15],
                                             [0.1, 0.1, 0.1], False)[1])

        # Spawning crosswalks
        # self.crosswalk = QLabsCrosswalk(self.qlabs)
        self.crosswalks = []
        self.crosswalks.append(
            self.crosswalk_qvl.spawn_degrees(
                location=[-2 + self.x_offset, -1.475 + self.y_offset, 0.01],
                rotation=[0, 0, 0],
                scale=[0.1, 0.1, 0.075],
                configuration=0
            )[1]
        )

        # traffic light
        self.traffic_light1_qvl.spawn_degrees([2.3 + self.x_offset, self.y_offset, 0], [0, 0, 0], scale=[.1, .1, .1],
                                              configuration=0, waitForConfirmation=True)
        self.traffic_light1_qvl.set_state(QLabsTrafficLight.STATE_GREEN)
        self.traffic_light2_qvl.spawn_degrees([-2.3 + self.x_offset, -1 + self.y_offset, 0], [0, 0, 180],
                                              scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
        self.traffic_light2_qvl.set_state(QLabsTrafficLight.STATE_RED)

        # mySpline = QLabsBasicShape(self.qlabs)
        self.shape_qvl.spawn_degrees([2.05 + self.x_offset, -1.5 + self.y_offset, 0.01], [0, 0, 0], [0.27, 0.02, 0.001],
                                     False)
        self.shape_qvl.spawn_degrees([-2.075 + self.x_offset, self.y_offset, 0.01], [0, 0, 0], [0.27, 0.02, 0.001],
                                     False)
        # self.shape_qvl.spawn_degrees([2.1 + self.x_offset, 1.5 + self.y_offset, 0.01], [0, 0, 0], [0.27, 0.02, 0.001], False)
        # self.shape_qvl.spawn_degrees([-1.3 + self.x_offset, 2.75 + self.y_offset, 0.01], [0, 0, -90], [0.27, 0.02, 0.001], False)

        # stopsign and traffic light regions
        self.stopsign_regions = np.stack([
            np.array([[2.1 + self.x_offset - (0.25 / 2), 1.15 + self.y_offset - (0.45 / 2)],
                      [2.1 + self.x_offset + (0.25 / 2), 1.15 + self.y_offset + (0.45 / 2)]], dtype=np.float32),
            np.array([[-0.95 + self.x_offset - (0.45 / 2), 2.75 + self.y_offset - (0.25 / 2)],
                      [-0.95 + self.x_offset + (0.45 / 2), 2.75 + self.y_offset + (0.25 / 2)]], dtype=np.float32)
        ])
        self.trafficstop_regions = np.stack([
            np.array([[-2.075 + self.x_offset - (0.45 / 2), 0.35 + self.y_offset - (0.25 / 2)],
                      [-2.075 + self.x_offset + (0.45 / 2), 0.35 + self.y_offset + (0.25 / 2)]], dtype=np.float32),
            np.array([[2.1 + self.x_offset - (0.25 / 2), -1.85 + self.y_offset - (0.45 / 2)],
                      [2.1 + self.x_offset + (0.25 / 2), -1.85 + self.y_offset + (0.45 / 2)]], dtype=np.float32)
        ])

        # stop regions
        # self.shape_qvl.spawn_degrees([2.1 + self.x_offset, 1.15 + self.y_offset, 0.01], [0, 0, 0], [0.25, 0.45, 0.001], False)
        # self.shape_qvl.spawn_degrees([-0.95 + self.x_offset, 2.75 + self.y_offset, 0.01], [0, 0, 0], [0.45, 0.25, 0.001], False)
        # self.shape_qvl.spawn_degrees([-2.075 + self.x_offset, 0.35 + self.y_offset, 0.01], [0, 0, 0], [0.25, 0.45, 0.001], False)
        # self.shape_qvl.spawn_degrees([2.1 + self.x_offset, -1.85 + self.y_offset, 0.01], [0, 0, 0], [0.25, 0.45, 0.001], False)

        # Start spawn model
        QLabsRealTime().start_real_time_model(rtmodels.QCAR_STUDIO)

    def traffic_light_control(self):
        # toggle traffic lights every 5 seconds
        if self.ep_steps % int(5 / self.dt) == 0:
            self.traffic_light_states[0] = "GREEN" if self.traffic_light_states[0] == "RED" else "RED"
            self.traffic_light_states[1] = "GREEN" if self.traffic_light_states[1] == "RED" else "RED"
            self.traffic_light1_qvl.set_state(
                QLabsTrafficLight.STATE_GREEN if self.traffic_light_states[0] == "RED" else QLabsTrafficLight.STATE_RED)
            self.traffic_light2_qvl.set_state(
                QLabsTrafficLight.STATE_GREEN if self.traffic_light_states[1] == "RED" else QLabsTrafficLight.STATE_RED)

    def get_in_trafficstop_region(self):
        x, y = self.state[:2]

        # check if in a traffic stop region
        in_trafficstop_region = False
        for i in self.traffic_light_states:
            if x >= self.trafficstop_regions[i, 0, 0] and x <= self.trafficstop_regions[i, 1, 0] and \
                    y >= self.trafficstop_regions[i, 0, 1] and y <= self.trafficstop_regions[i, 1, 1] and \
                    self.traffic_light_states[i] == "RED":
                in_trafficstop_region = True

        return in_trafficstop_region

    def get_in_stopsign_region(self):
        x, y = self.state[:2]

        # check if in a stop region
        in_stopsign_region = False
        for i in range(self.stopsign_regions.shape[0]):
            if x >= self.stopsign_regions[i, 0, 0] and x <= self.stopsign_regions[i, 1, 0] and y >= \
                    self.stopsign_regions[i, 0, 1] and y <= self.stopsign_regions[i, 1, 1]:
                in_stopsign_region = True

        return in_stopsign_region

    def get_state_duration(self):
        return self.FSM.duration

    def get_state(self, class_id, actor_number):
        """
            Gets state (by default gets ground truth)
        """
        c = CommModularContainer()
        c.classID = class_id  # self.car_qvl.classID
        c.actorNumber = actor_number  # self.car_qvl.actorNumber
        c.actorFunction = QLabsActor.FCN_REQUEST_WORLD_TRANSFORM
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if self.qlabs.send_container(c):
            c = self.qlabs.wait_for_container(class_id, actor_number, QLabsActor.FCN_RESPONSE_WORLD_TRANSFORM)
            x, y, z, roll, pitch, yaw, sx, sy, sz, = struct.unpack(">fffffffff", c.payload[0:36])

            # calc velocity
            vx = (x - self.state[0]) / self.dt
            vy = (y - self.state[1]) / self.dt
            v = np.hypot(vx, vy)

            # calc rate of turn
            w = (yaw - self.state[2]) / self.dt

            # calc acceleration
            a = (v - self.state[3]) / self.dt

            state = np.array([x, y, yaw, v, w, a])
            self.state = state

            return state
        else:
            raise Exception("Could not get GT State!")

    def step(self, action, metrics):
        # placeholders
        obs = {}
        reward = 0.0
        done = self.ep_steps >= self.max_ep_steps
        info = {}

        # start timer for profiling
        t = time.perf_counter()

        # Condition action on FSM
        # if self.FSM.state in [1, 2]:
        #    action[0] = 0.0

        # do something with action
        self.car.read_write_std(action[0], action[1])
        time.sleep(self.dt)
        self.last_action = time.perf_counter()

        # update world
        self.traffic_light_control()

        # get privileged information (Very Slow! Limits to 18 Hz!)
        if self.privileged:
            # GT state
            ego_state = self.get_state(self.car_qvl.classID, self.car_qvl.actorNumber)
            orig = ego_state[:2]
            yaw = -ego_state[2]
            rot = np.array([[np.cos(yaw), np.sin(yaw)],
                            [-np.sin(yaw), np.cos(yaw)]])

            # GT waypoints
            norm_dist = np.linalg.norm(self.next_waypoints[:self.max_lookahead_indices] - ego_state[:2], axis=1)
            dist_ix = np.argmin(norm_dist)
            delta_ix = dist_ix + self.prev_dist_ix
            self.prev_dist_ix = dist_ix
            self.next_waypoints = self.next_waypoints[dist_ix:]

            # add first waypoints if at end
            if self.next_waypoints.shape[0] < self.max_lookahead_indices:
                slop = self.max_lookahead_indices - self.next_waypoints.shape[0]
                self.next_waypoints = np.concatenate([self.next_waypoints, self.waypoint_sequence[:slop]])

        # get sensor data and update observation
        # rgbd_image = self.rgbd.get_image()
        csi_front, _, _, _ = self.csi.get_images()

        # print("CSI: {}".format(time.perf_counter() - t))
        # cv2.imshow("RGB", csi_front)
        # cv2.waitKey(1)

        obs["state"] = ego_state if self.privileged else None
        obs["waypoints"] = np.matmul(self.next_waypoints[:self.max_lookahead_indices] - orig,
                                     rot) if self.privileged else None
        obs["image"] = cv2.resize(csi_front[:, :, :3], (160, 120))
        # obs["image"] = csi_front

        # render waypoints
        # pred_waypoints = np.matmul(metrics["waypoints"], rot.T) + orig
        # for i in range(0, pred_waypoints.shape[0], 5):
        #    location = [pred_waypoints[i, 0], pred_waypoints[i, 1], 0] 
        #    self.basicshape.spawn_id(i, location=location, rotation=[0, 0, 0], scale=[0.01, 0.01, 0.02], configuration=1, waitForConfirmation=False)

        # check if in stopsign or traffic light region
        self.in_stopsign_region = self.get_in_stopsign_region()
        self.in_trafficstop_region = self.get_in_trafficstop_region()

        # get reward for performing action
        fsm_penalty = self.FSM.step()  # check if we get a penalty on FSM state change (traffic violation)
        deviation_penalty = 5.0 if norm_dist[dist_ix] >= 0.25 else 0.0  # check if we are off the road
        deviation_penalty = max(0.0, 10 * (norm_dist[dist_ix] - 0.1))
        reward = deviation_penalty  # 100*(delta_ix / self.waypoint_sequence.shape[0]) - fsm_penalty #+ deviation_penalty

        if self.privileged and (
                np.linalg.norm(self.goal - ego_state[:2]) < self.goal_thresh and len(self.next_waypoints) < 50):
            done = True

        if self.privileged and norm_dist[dist_ix] >= 0.25:
            done = True
            reward -= 30.0

        # if done then stop car so I don't get dizzy
        if done:
            self.car.read_write_std(0.0, 0.0)

        self.ep_steps += 1
        return obs, reward, done, info

    def reset(self):
        # reload env
        self.initialize_environment()

        # placeholders
        obs = {}
        reward = 0.0
        done = False
        info = {}

        self.prev_dist_ix = 0

        # initialize episode parameters
        self.ep_steps = 0
        self.last_action = time.perf_counter()

        # select random waypoint sequence
        sequence = self.sequences[np.random.choice(list(self.sequences.keys()))]
        '''sequence = None
        while not sequence:
            try:
                start_id = np.random.randint(len(self.roadmap.nodes))
                sequence = self.roadmap.generate_random_cycle(start_id)
            except ValueError as e:
                print("Could not find cycles for starting id: {}".format(start_id))'''

        self.waypoint_sequence = self.roadmap.generate_path(sequence)
        self.next_waypoints = self.waypoint_sequence
        self.goal = self.waypoint_sequence[-1]  # x, y coords of goal

        # get spawn position and orientation
        dpos = self.waypoint_sequence[5] - self.waypoint_sequence[0]
        theta = np.arctan2(dpos[1], dpos[0])  # np.arccos(np.dot(dpos, np.array([1, 0])) / np.linalg.norm(dpos))
        position = [self.waypoint_sequence[0, 0], self.waypoint_sequence[0, 1], 0.005]
        # position = [-0.75, -0.90, 0.005]
        orientation = [0, 0, theta]

        # call setup function or spawn car
        # self.car_qvl = setup(position, orientation)
        # recreate car object
        self.car_qvl = QLabsQCar(self.qlabs)
        status = self.car_qvl.spawn_id(actorNumber=0, location=position, rotation=orientation, scale=[.1, .1, .1],
                                       configuration=0, waitForConfirmation=True)
        self.car_qvl.possess(6)  # 6 = overhead camera
        self.car = QCar(readMode=0)
        self.in_stopsign_region = False
        self.in_trafficstop_region = False
        # self.rgbd = VirtualRGBDCamera(apply_lighting_noise=False)
        self.csi = VirtualCSICamera(enable_front=True, image_shape=(640, 480))

        # render waypoints
        # for i in range(0, self.waypoint_sequence.shape[0], 5):
        #    location = [self.waypoint_sequence[i, 0], self.waypoint_sequence[i, 1], 0] 
        #    self.basicshape.spawn_id(i, location=location, rotation=[0, 0, 0], scale=[0.01, 0.01, 0.02], configuration=1, waitForConfirmation=False)

        # get sensor data and update observation
        # rgbd_image = self.rgbd.get_image()
        csi_front, _, _, _ = self.csi.get_images()
        ego_state = self.get_state(self.car_qvl.classID, self.car_qvl.actorNumber)
        orig = ego_state[:2]
        yaw = -ego_state[2]
        rot = np.array([[np.cos(yaw), np.sin(yaw)],
                        [-np.sin(yaw), np.cos(yaw)]])

        # get initial observation
        obs["state"] = ego_state
        obs["waypoints"] = np.matmul(self.next_waypoints[:self.max_lookahead_indices] - orig,  # np.matmul(a, b): matrix a Matrix Multiplication b, get a tensor
                                     rot) if self.privileged else None
        obs["image"] = cv2.resize(csi_front[:, :, :3], (160, 120))
        # obs["image"] = csi_front

        return obs, reward, done, info


if __name__ == "__main__":
    from core.policies.pure_pursuit import PurePursuitPolicy

    sim = QLabsSim(dt=0.05)
    num_episodes = 100
    policy = PurePursuitPolicy(max_lookahead_distance=0.75)

    for ep in range(num_episodes):
        obs, reward, done, info = sim.reset()
        while not done:
            action, _ = policy(obs)
            obs, reward, done, info = sim.step(action)

# env = QLabsSim()
# obs, _, _, _ = env.reset()
# print("Observation space shape:", obs.shape)
