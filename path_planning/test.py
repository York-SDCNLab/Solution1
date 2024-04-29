import sys
import math 

from qvl.basic_shape import QLabsBasicShape
from qvl.qlabs import QuanserInteractiveLabs 
from scripts.Setup_Competition import setup

from acc_roadmap import ACCRoadMap 

if __name__ == "__main__": 
    # generate acc map 
    x_offset = 0.13
    y_offset = 1.67
    setup(initialPosition=[-1.335+ x_offset, -2.5+ y_offset, 0.005], initialOrientation=[0,0,-math.pi/4])
    # connect to qlab 
    qlabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    # set up 
    roadmap = ACCRoadMap() 
    basic_shape = QLabsBasicShape(qlabs=qlabs, verbose=False) 
    node_sequence = [0, 2, 4, 13, 19, 17, 15, 6, 0] 
    waypoint_sequence = roadmap.generate_path(node_sequence) 

    # draw road points 
    rotation = [0, 0, 0] 
    scale = [0.01, 0.01, 0.02]
    for i in range(len(waypoint_sequence[0])): 
        location = [waypoint_sequence[0][i], waypoint_sequence[1][i], 0] 
        basic_shape.spawn_id(i, location=location, rotation=rotation, scale=scale, configuration=1, waitForConfirmation=False) 

    qlabs.close() 