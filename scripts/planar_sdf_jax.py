import sys
import os
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
build_dir = os.path.abspath(os.path.join(root_dir, 'build'))
planarmap_dir = os.path.abspath(os.path.join(os.path.join(root_dir, 'map'),'planar'))

sys.path.append(build_dir)
import libplanar_sdf
from generate_sdf_2d import *

from jax import numpy as jnp
from jax import grad
import jax
import numpy as np


import jax
import jax.numpy as np
from jax import grad

# Define a function that involves rounding and matrix indexing
def my_function(x, matrix):
    # Round to the nearest integer
    rounded_index = jnp.round(x)
    
    # Convert to integer for indexing
    index = jnp.int_(rounded_index)
    
    # Access the matrix element at the rounded index
    y = matrix[index]
    
    return y

# Use JAX grad to compute the gradient
grad_my_function = grad(my_function, argnums=0)  # Compute gradient with respect to the first argument (x)

# Input values
x = 3.14
matrix = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Replace this with your matrix

# Compute the gradient
gradient_value = grad_my_function(x, matrix)

print("Input value:", x)
print("Gradient:", gradient_value)

# # 2d sdf
# sdf_2d, map2d = generate_2dsdf("SingleObstacleMap")
# field_data = map2d.get_field()
# cell_size = map2d.cell_size()
# width = map2d.width()
# height = map2d.height()
    
# def sdf(point, field):
#     x = point[0]
#     y = point[1]
    
#     rf = y / cell_size
#     cf = x / cell_size
    
#     print("rf, cf")
#     print(rf, cf)
    
#     rounded_rf = jax.lax.floor(rf)
#     rounded_cf = jax.lax.floor(cf)
    
#     # Use jax.ops.index to create an integer index
#     ri = jnp.int_(rounded_rf)
#     ci = jnp.int_(rounded_cf)
    
#     print("ri, ci")
#     print(ri, ci)
    
#     return field[ri, ci]
 
# t_pt1 = np.ones(2, dtype=np.float64)
# dist = sdf_2d.getSignedDistance(t_pt1)
# gradient_pt1 = sdf_2d.getGradient(t_pt1)
       
# dist_sdf = sdf(t_pt1, field_data)

# print("dist")
# print(dist)
# print("dist_sdf")
# print(dist_sdf)

# print("gradient_pt1")
# print(gradient_pt1)
# print("grad dist_sdf")
# print(grad(sdf, 0)(t_pt1, field_data))
