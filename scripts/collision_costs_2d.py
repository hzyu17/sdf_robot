## Collision costs definitions and their derivatives using a SDF
# Hongzhe Yu, 12/19/2023

import sys
import os
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
build_dir = os.path.abspath(os.path.join(root_dir, 'build'))
src_dir = os.path.abspath(os.path.join(root_dir, 'src'))
planarmap_dir = os.path.abspath(os.path.join(os.path.join(root_dir, 'map'),'planar'))

print("build_dir")
print(build_dir)

sys.path.append(build_dir)
sys.path.append(planarmap_dir)

import libplanar_sdf
import numpy as np
import plotly.graph_objects as go
from generate_map import *

# hinge(dist), (\par h)/(\par dist)
def hinge_loss_gradient(dist, eps_obs, slope=1):
    if dist > eps_obs:
        return 0.0, 0.0
    if dist <= eps_obs:
        return dist, -slope
    
# hinge(sdf(pt)), (\par h)/(\par pt)
def sdf_loss_gradient(pt, sdf_2d, eps_obs, slope=1):
    dist, g_sdf = sdf_2d.getSignedDistance(pt), sdf_2d.getGradient(pt)    
    h, g_h = hinge_loss_gradient(dist, eps_obs, slope)
    return h, g_h*g_sdf

## \| hinge(pt) \|^2_{sig_obs}
def collision_cost_gradient(sig_obs, vec_pts, sdf_2d, eps_obs, slope=1):
    n_pts, pt_dim = vec_pts.shape
    vec_hinge = np.zeros(n_pts, dtype=np.float32)
    g_vec_hinge = np.zeros((n_pts, pt_dim), dtype=np.float32)
    g_vec_pts = np.zeros(pt_dim, dtype=np.float32)
    
    for r in range(n_pts):
        pt = vec_pts[r]
        
        h, g_h_pt = sdf_loss_gradient(pt, sdf_2d, eps_obs, slope)
        vec_hinge[r] = h
        g_vec_hinge[r] = g_h_pt
        
    Sig_obs = sig_obs * np.eye(n_pts, dtype=np.float32)
    collision_cost = vec_hinge.T @ Sig_obs @ vec_hinge
    g_vec_pts = 2*Sig_obs@vec_hinge@g_vec_hinge
    
    return collision_cost, g_vec_pts
    
    
def generate_2dsdf(map_name):
    origin = np.array([0.0, 0.0], dtype=np.float32)
    cell_size = 0.1
    data = generate_map("SingleObstacleMap")
    
    print("data.shape")
    print(data.shape)
    planar_sdf = libplanar_sdf.PlanarSDF(origin, cell_size, data)

    return planar_sdf


if __name__ == '__main__':
   
    ## Planar sdf
    map_name = "SingleObstacleMap"
    planar_sdf = generate_2dsdf(map_name)
    
    