## Test the planar signed distance field
# Hongzhe Yu, 12/20/2023

import sys
import os
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
script_dir = os.path.abspath(os.path.join(root_dir, 'scripts'))

sys.path.append(script_dir)

from generate_sdf_2d import *
from collision_costs_2d import *
import numpy as np

def test_signed_distance():
    sdf_2d = generate_2dsdf("SingleObstacleMap", False)
    # obstacle range: (x: [22.0, 32.0]; y: [10.0, 16.0])
    
    # test point: the origin ([0, 0])
    nearest_dist = np.sqrt(220.0*220.0+100.0*100.0)*0.1
    atan_gradient = np.arctan2(-10.0, -22.0)
    
    t_pt1 = np.zeros(2, dtype=np.float64)
    dist = sdf_2d.getSignedDistance(t_pt1)
    gradient_pt1 = sdf_2d.getGradient(t_pt1)
    
    assert(abs(dist - nearest_dist) < 1e-6)
    assert(abs(np.arctan2(gradient_pt1[1], gradient_pt1[0]) - atan_gradient) < 1e-2)
    
    eps_obs = 0.1
    slope = 1
    h, g = sdf_loss_gradient(t_pt1, sdf_2d, eps_obs, slope)
    
    assert(abs(h - 0.0) < 1e-6)
    assert(np.linalg.norm(g-np.zeros(2, dtype=np.float32)) < 1e-2)
    
    # test hinge loss and gradient
    t_pt2 = np.array([22.0, 9.9], dtype=np.float64)
    g_nearest = np.array([0, 1], dtype=np.float32)
    
    h, g = sdf_loss_gradient(t_pt2, sdf_2d, eps_obs, slope)
    
    assert(abs(h - 0.1) < 1e-6)
    assert(np.linalg.norm(g/np.linalg.norm(g) - g_nearest) < 1e-2)
    
    # test slope
    slope = 2
    h, g = sdf_loss_gradient(t_pt2, sdf_2d, eps_obs, slope)
    
    assert(abs(h - 0.2) < 1e-6)
    assert(np.linalg.norm(g/np.linalg.norm(g) - g_nearest) < 1e-2)
    
    # test collision loss gradient 
    pts = np.zeros((2, 2), dtype=np.float32)
    pts[0] = t_pt1
    pts[1] = t_pt2
    
    
    
if __name__ == '__main__':
    test_signed_distance()