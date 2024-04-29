import sys 
import heapq
import numpy as np 

from qvl.qlabs import QuanserInteractiveLabs 
from hal.utilities.path_planning import RoadMap
from qvl.basic_shape import QLabsBasicShape

sys.path.append('path_planning/')
from path_planning.constants import X_OFFSET 
from path_planning.constants import Y_OFFSET 
from path_planning.constants import ACC_SCALE 
from path_planning.constants import NODE_POSES_RIGHT_COMMON
from path_planning.constants import NODE_POSES_RIGHT_LARGE_MAP 
from path_planning.constants import EDGE_CONFIGS_RIGHT_COMMON 
from path_planning.constants import EDGE_CONFIGS_RIGHT_LARGE_MAP 
from path_planning.constants import WAYPOINT_ROTATION
from path_planning.constants import WAYPOINT_SCALE 

class ACCRoadMap(RoadMap): 
    def __init__(self): 
        # parent class initialization 
        super().__init__()
        # read nodes and edges 
        node_positions = NODE_POSES_RIGHT_COMMON + NODE_POSES_RIGHT_LARGE_MAP 
        edges = EDGE_CONFIGS_RIGHT_COMMON + EDGE_CONFIGS_RIGHT_LARGE_MAP  
        # add scaled nodes to acc map 
        for position in node_positions: 
            position[0] = ACC_SCALE * (position[0] - X_OFFSET) 
            position[1] = ACC_SCALE * (Y_OFFSET - position[1]) 
            self.add_node(position) 
        # add scaled edge to acc map 
        for edge in edges: 
            edge[2] = edge[2] * ACC_SCALE 
            self.add_edge(*edge) 

    #generate a random cycle from a given starting node
    def generate_random_cycle(self, start, min_length=3):
        #depth first search for finding all cycles that start and end at the starting point
        def dfs(start):
            fringe = [(start, [])]

            while fringe:
                node, path = fringe.pop()
                if path and node == start:
                    yield path
                    continue
                for next_edges in node.outEdges:
                    next_node = next_edges.toNode
                    if next_node in path:
                        continue
                    fringe.append((next_node, path + [next_node]))

        start_node = self.nodes[start]
        cycles = [[start_node] + path for path in dfs(start_node) if len(path) > min_length]
        num_cycles = len(cycles)

        return cycles[np.random.randint(num_cycles)]

    #wrap as numpy array object
    def generate_path(self, sequence):
        if type(sequence) == np.ndarray:
            sequence = sequence.tolist()

        return np.array(super().generate_path(sequence)).transpose(1, 0) #[N, (x, y)]