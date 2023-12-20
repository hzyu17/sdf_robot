## Test the planar signed distance field
# Hongzhe Yu, 12/20/2023

import sys
import os
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
script_dir = os.path.abspath(os.path.join(root_dir, 'scripts'))

sys.path.append(script_dir)

from generate_sdf_2d import *

planar_sdf = generate_2dsdf("SingleObstacleMap", False)
