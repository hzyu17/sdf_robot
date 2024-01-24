## Visualize and verify the implementation of sdf and its gradient field
# Hongzhe Yu

from collision_costs_2d import *
import plotly.graph_objects as go

if __name__ == '__main__':
    sdf_2d, planarmap = generate_2dsdf("SingleObstacleMap", False)
    origin = planarmap.get_origin()
    width = planarmap.get_width() * planarmap.get_cell_size()
    height = planarmap.get_height() * planarmap.get_cell_size()
    
    x_lim = [origin[1], origin[1]+width]
    y_lim = [origin[0], origin[0]+height]
    
    x_grid = np.linspace(x_lim[0], x_lim[1], 50)
    y_grid = np.linspace(y_lim[0], y_lim[1], 50)
    
    X, Y = np.meshgrid(x_grid, y_grid)
    
    print(X.shape)
    
    eps_obs = 0.2
    
    hingeloss_grid = np.zeros_like(X, dtype=np.float64)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pt_i = np.array([[X[i, j]], [Y[i, j]]], dtype=np.float64)
            hingeloss_grid[i, j], _ = hinge_sdf_loss_gradient(pt_i, sdf_2d, eps_obs)
    
    # Plot hinge loss surface and gradient field
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=hingeloss_grid, opacity=0.5)])

    fig.update_layout(title='Covariance Steering for a Linear System', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.update_layout(scene=dict(
                      xaxis_title='X',
                      yaxis_title='Y',
                      zaxis_title='Time'
                    ))
        
    # Show image
    fig.show()
    