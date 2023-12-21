## Generate planar map 
# 12/20/2023

import sys
import os
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
build_dir = os.path.abspath(os.path.join(root_dir, 'build'))
planarmap_dir = os.path.abspath(os.path.join(os.path.join(root_dir, 'map'),'planar'))

sys.path.append(build_dir)

import libplanar_sdf
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
    
    ## The input coordinate (x, y) follows the conventional right-hand coordinate system definitions,
    #  the bottom-left of a field matrix is the origin. Notice that in the PlanarSDF class definitions, 
    #  the directions are inversed and needs a transform from the RH to the field data matrix.
    def add_box_xy(self, xmin, ymin, shape):
        xmax = xmin + shape[0]
        ymax = ymin + shape[1]
        xmin_indx, xmax_indx = int(xmin / self.cell_size), int(xmax / self.cell_size)
        ymin_indx, ymax_indx = int(ymin / self.cell_size), int(ymax / self.cell_size)
        
        self.map[ymin_indx:ymax_indx, xmin_indx:xmax_indx] = 1.0
        
        self.generate_SDField()

    def add_box_index(self, xmin, ymin, shape):
        xmax = xmin + shape[0]
        ymax = ymin + shape[1]
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
    
    def get_field(self):
        return self.field
    
    def draw_map(self):    
        fig, ax = plt.subplots()

        cmap = plt.cm.colors.ListedColormap(['white', 'black'])

        # Create a heatmap
        im = ax.imshow(self.map, cmap=cmap, interpolation='nearest', origin='lower',
                   extent=[0, self.map_width*self.cell_size, 0, self.map_height*self.cell_size])
        
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # num_rows, num_cols = binary_matrix.shape
        plt.xlim([0, self.map_width * self.cell_size])
        plt.ylim([0, self.map_height * self.cell_size])
        
        plt.title('Obstacle environment')
        
        # plt.show()
        
        return fig, ax

    
def generate_planarmap(map_name, cell_size, save_map=False):
    origin = np.array([0.0, 0.0], dtype=np.float64)
    width = 400
    height = 300
    
    # map range: (x: [0.0, 50.0]; y: [0.0, 50.0])
    m = PlanarMap(origin, cell_size, width, height)
    
    if map_name == "SingleObstacleMap":
        # obstacle range: (x: [10.0, 20.0]; y: [10.0, 16.0])
        m.add_box_xy(10.0, 10.0, [10.0, 6.0])
        
    if save_map:
        m.save_map(planarmap_dir + '/' + map_name + '.csv', 
                   planarmap_dir + '/' + map_name + '_field' + '.csv')
    
    return m


def generate_2dsdf(map_name="SingleObstacleMap", savemap=False):
    origin = np.array([0.0, 0.0], dtype=np.float64)
    cell_size = 0.1
    planarmap = generate_planarmap(map_name, cell_size, savemap)
    field_data = planarmap.get_field()
    
    planar_sdf = libplanar_sdf.PlanarSDF(origin, cell_size, field_data)

    return planar_sdf, planarmap