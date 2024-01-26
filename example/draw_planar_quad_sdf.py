import numpy as np
import matplotlib.pyplot as plt

import sys
import os

file_path = os.path.abspath(__file__)
example_dir = os.path.dirname(file_path)

py_dir = os.path.abspath(os.path.join(example_dir, '..'))
scripts_dir =os.path.abspath(os.path.join(py_dir, 'scripts'))
sys.path.append(scripts_dir)

from generate_sdf_2d import * 
from matplotlib.patches import Rectangle, Arrow

# Defines the forward kinematics for collision-checking balls and their gradients to the states.  
def vec_balls(x, L, n_balls, radius):
    v_pts = np.zeros((n_balls, 2), dtype=np.float64)
    v_g_states = np.zeros((n_balls, x.shape[0]), dtype=np.float64)
    v_radius = radius * np.ones(n_balls, dtype=np.float64)
    pos_x = x[0]
    pos_z = x[1]
    phi = x[2]
    
    l_pt_x = pos_x - (L-radius*1.5)*np.cos(phi)/2.0
    l_pt_z = pos_z - (L-radius*1.5)*np.sin(phi)/2.0
    
    for i in range(n_balls):
        pt_xi = l_pt_x + L*np.cos(phi)/n_balls*i
        pt_zi = l_pt_z + L*np.sin(phi)/n_balls*i
        v_pts[i] = np.array([pt_xi, pt_zi])
        v_g_states[i] = np.array([1.0, 1.0, -L*np.sin(phi)/n_balls, 0.0, 0.0, 0.0])
    
    return v_pts, v_g_states, v_radius

def compute_pqud_patches(x, L, H, patch_color='b'):
    center_location = x[0:2]
    theta = x[2]
    
    center1 = center_location[0]
    center2 = center_location[1]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    X = np.array([-L/2, L/2, L/2, -L/2], dtype=np.float64)
    Y = np.array([-H/2, -H/2, H/2, H/2], dtype=np.float64)

    T = np.zeros((2, 4), dtype=np.float64)
    for i in range(4):
        T[:, i] = np.dot(R, np.array([X[i], Y[i]]))

    x_lower_left = center1 + T[0, 0]
    y_lower_left = center2 + T[1, 0]
    rect = Rectangle((x_lower_left, y_lower_left), L, H, angle=theta/3.14*180, edgecolor='none', facecolor=patch_color)
    
    arrow_width = 0.1
    # Plot 2 arrows representing the inputs
    # left arrow
    l_arrow_x = center1 - (L-arrow_width)/2*np.cos(theta)
    l_arrow_z = center2 - (L-arrow_width)/2*np.sin(theta) + H/2*np.cos(theta)
    l_arrow = Arrow(l_arrow_x, l_arrow_z, -np.sin(theta), np.cos(theta), width=0.6, edgecolor='none', facecolor='green')
    
    # right arrow
    r_arrow_x = center1 + (L-4*arrow_width)/2*np.cos(theta)
    r_arrow_z = center2 + (L-4*arrow_width)/2*np.sin(theta) + H/2*np.cos(theta)
    r_arrow = Arrow(r_arrow_x, r_arrow_z, -np.sin(theta), np.cos(theta), width=0.6, edgecolor='none', facecolor='green')
    
    n_balls = 5
    radius = L/7.0
    v_pts, _, v_radius = vec_balls(x, L, n_balls, radius)
    v_circles = []
    for i in range(n_balls):
        v_circles.append(plt.Circle(v_pts[i], v_radius[i], edgecolor='r', facecolor='none', linewidth=2))
        
    return rect, l_arrow, r_arrow, v_circles

# @njit
def draw_quad_balls(x, L, H, fig, ax, draw_ball=True, patch_color='b'):
    rect, l_arrow, r_arrow, v_circles = compute_pqud_patches(x, L, H, patch_color)
    
    ax.add_patch(l_arrow)
    ax.add_patch(r_arrow)
    ax.add_patch(rect)
    
    if draw_ball:
        for i in range(len(v_circles)):
            ax.add_patch(v_circles[i])
            plt.axis('equal')
            
    return fig, ax
    
if __name__ == '__main__':
    L = 5.0
    H = 0.35
    state = np.array([25.0, 15.0, np.pi/6, 0.0, 0.0, 0.0])
    sdf_2d, planarmap = generate_2dsdf("SingleObstacleMap", False)

    fig, ax = plt.subplots()
    
    fig, ax = planarmap.draw_map(fig, ax, plot=True)
    fig, ax = draw_quad_balls(state, L, H, fig, ax, True, 'b')
    
    fig.savefig(example_dir+"/example_quad2d.png", dpi=500, bbox_inches='tight')