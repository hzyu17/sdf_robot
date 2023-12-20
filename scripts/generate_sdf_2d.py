## Generate planar map 
# 12/20/2023

import sys
import os
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
build_dir = os.path.abspath(os.path.join(root_dir, 'build'))
planarmap_dir = os.path.abspath(os.path.join(os.path.join(root_dir, 'map'),'planar'))

sys.path.append(build_dir)

import libplanar_sdf
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import numpy as np

class PlanarMap:
    def __init__(self, origin, cell_size, map_width, map_height, map_name='defaultmap') -> None:
        self.map_name = map_name
        self.origin = origin
        self.cell_size = cell_size
        self.map_width = map_width
        self.map_height = map_height
        self.map = np.zeros((map_width, map_height), dtype=np.float64)
        self.field = np.zeros((map_width, map_height), dtype=np.float64)

    def update_name(self, map_name):
        self.map_name = map_name
    
    def add_box_xy(self, xmin, ymin, size):
        xmax = xmin + size[0]
        ymax = ymin + size[1]
        xmin_indx, xmax_indx = int(xmin // self.cell_size), int(xmax // self.cell_size)
        ymin_indx, ymax_indx = int(ymin // self.cell_size), int(ymax // self.cell_size)
        
        self.map[xmin_indx:xmax_indx, ymin_indx:ymax_indx] = 1.0
        
        self.generate_SDField()

    def add_box_index(self, xmin, ymin, size):
        xmax = xmin + size[0]
        ymax = ymin + size[1]
        self.map[xmin:xmax, ymin:ymax] = 1.0
        self.generate_SDField()
        
    def generate_SDField(self):
        inverse_map = 1.0 - self.map
        inside_dist = bwdist(self.map)
        outside_dist = bwdist(inverse_map)
        self.field = (outside_dist - inside_dist)*self.cell_size
    
    def save_map(self, map_name, field_name):
        np.savetxt(map_name, self.map, delimiter=',')
        np.savetxt(field_name, self.field, delimiter=',')
        
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
        m.save_map(planarmap_dir + '/' + map_name + '.csv', 
                   planarmap_dir + '/' + map_name + '_field' + '.csv')
    
    return m.get_map()


def generate_2dsdf(map_name="SingleObstacleMap", savemap=False):
    origin = np.array([0.0, 0.0], dtype=np.float32)
    cell_size = 0.1
    data = generate_map(map_name, savemap)
    
    planar_sdf = libplanar_sdf.PlanarSDF(origin, cell_size, data)

    return planar_sdf