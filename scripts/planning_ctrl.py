# region: package imports
import os
import numpy as np
import random
import sys
import time
import math
import struct
import cv2
import random

# environment objects

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.basic_shape import QLabsBasicShape
from qvl.system import QLabsSystem
from qvl.walls import QLabsWalls
from qvl.flooring import QLabsFlooring
from qvl.stop_sign import QLabsStopSign
from qvl.crosswalk import QLabsCrosswalk
import pal.resources.rtmodels as rtmodels

from Setup_Competition import setup



if __name__ == '__main__':
    #setup(initialPosition=[-1.335+ x_offset, -2.5+ y_offset, 0.005], initialOrientation=[0,0,-math.pi/4])
    setup(initialPosition=[0, 1, 0.005], initialOrientation=[0,0,0])
