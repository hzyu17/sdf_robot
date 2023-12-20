## Test the planar signed distance field
# Hongzhe Yu, 12/20/2023

import sys
import os
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
script_dir = os.path.abspath(os.path.join(root_dir, 'scripts'))

sys.path.append(script_dir)

from generate_sdf_2d import *
import numpy as np

def test_signeddistance():
    sdf_2d = generate_2dsdf("SingleObstacleMap", False)
    
    # test point: the origin ([0, 0])
    nearest_dist = np.sqrt(220.0*220.0+100.0*100.0)*0.1
    atan_gradient = np.arctan2(-10.0, -22.0)
    
    t_pt = np.zeros(2, dtype=np.float64)
    dist = sdf_2d.getSignedDistance(t_pt)
    gradient_pt = sdf_2d.getGradient(t_pt)
    
    assert(abs(dist - nearest_dist) < 1e-6)
    assert(abs(np.arctan2(gradient_pt[1], gradient_pt[0]) - atan_gradient) < 1e-2)
    
    
if __name__ == '__main__':
    test_signeddistance()