## Generate planar map 
# 12/20/2023

import numpy as np

class PlanarMap:
    def __init__(self, origin, cell_size, map_width, map_height, map_name='defaultmap') -> None:
        self.map_name = map_name
        self.origin = origin
        self.cell_size = cell_size
        self.map_width = map_width
        self.map_height = map_height
        self.map = np.zeros((map_width, map_height), dtype=np.float32)

    def update_name(self, map_name):
        self.map_name = map_name
    
    def add_box_xy(self, xmin, ymin, size):
        xmax = xmin + size[0]
        ymax = ymin + size[1]
        xmin_indx, xmax_indx = int(xmin // self.cell_size), int(xmax // self.cell_size)
        ymin_indx, ymax_indx = int(ymin // self.cell_size), int(ymax // self.cell_size)
        
        self.map[xmin_indx:xmax_indx, ymin_indx:ymax_indx] = 1.0

    def add_box_index(self, xmin, ymin, size):
        xmax = xmin + size[0]
        ymax = ymin + size[1]
        self.map[xmin:xmax, ymin:ymax] = 1.0
    
    def save_map(self, filename):
        np.savetxt(filename, self.map, delimiter=',')
        
    def get_map(self):
        return self.map
    
    
def generate_map(map_name, save_map=False):
    origin = np.array([0.0, 0.0])
    cell_size = 0.1
    width = 500
    height = 500
    
    # map range: (x: [0.0, 50.0]; y: [0.0, 50.0])
    m = PlanarMap(origin, cell_size, width, height)
    
    if map_name == "SingleObstacleMap":
        # obstacle range: (x: [22.0, 32.0]; y: [10.0, 16.0])
        m.add_box_xy(22.0, 10.0, [10.0, 6.0])
        
    if save_map:
        m.save_map(map_name+'.csv')
    
    return m.get_map()
