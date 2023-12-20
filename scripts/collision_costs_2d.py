## Collision costs definitions and their derivatives using a SDF
# Hongzhe Yu, 12/19/2023

import numpy as np
import plotly.graph_objects as go
from generate_sdf_2d import *

# hinge(dist), (\par h)/(\par dist)
def hinge_loss_gradient(dist, eps_obs, slope=1):
    if dist > eps_obs:
        return 0.0, 0.0
    if dist <= eps_obs:
        return slope*dist, -slope
    
# hinge(sdf(pt)), (\par h)/(\par pt)
def hinge_sdf_loss_gradient(pt, sdf_2d, eps_obs, slope=1):
    dist, g_sdf = sdf_2d.getSignedDistance(pt), sdf_2d.getGradient(pt)    
    h, g_h = hinge_loss_gradient(dist, eps_obs, slope)
    return h, g_h*g_sdf

## \| hinge(pt) \|^2_{sig_obs}
def collision_lost_gradient(sig_obs, vec_pts, sdf_2d, eps_obs, slope=1):
    n_pts, pt_dim = vec_pts.shape
    vec_hinge = np.zeros(n_pts, dtype=np.float32)
    g_vec_hinge = np.zeros((n_pts, pt_dim), dtype=np.float32)
    g_vec_pts = np.zeros(pt_dim, dtype=np.float32)
    
    for r in range(n_pts):
        pt = vec_pts[r]
        
        h, g_h_pt = hinge_sdf_loss_gradient(pt, sdf_2d, eps_obs, slope)
        vec_hinge[r] = h
        g_vec_hinge[r] = g_h_pt
        
    Sig_obs = sig_obs * np.eye(n_pts, dtype=np.float32)
    collision_cost = vec_hinge.T @ Sig_obs @ vec_hinge
    g_vec_pts = 2*Sig_obs@vec_hinge@g_vec_hinge
    
    return collision_cost, g_vec_pts
    