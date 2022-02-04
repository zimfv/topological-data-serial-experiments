import matplotlib.pyplot as plt
import numpy as np

def draw_balls(data, radius, color='orchid', linewidth=0, alpha=0.5, circle_points=24, fill=True, ax=plt):
    """
    Plots balls arround points from data.
    
    Parameters:
    -----------
    data : np.ndarray or list of np.ndarrays
        List of points.
        
    radius : float
        Ball radius.
        
    color : color or rgba tuple
        
    linewidth : scalar
        
    alpha : float
        The alpha blending value, between 0 (transparent) and 1 (opaque).
            
    circle_points : int
        Number of points in circle. (If fill is True, then that will be number of ponts in top and bot arc)
        
    fill : bool
        Fill circle if True.
        
    ax : matplotlib.pyplot or Axes class
    """
    if fill:
        arc_x = radius*np.cos(np.pi*np.arange(circle_points+1)/circle_points)
        arc_y1 = radius*np.sin(np.pi*np.arange(circle_points+1)/circle_points)
        arc_y2 = radius*np.sin(-np.pi*np.arange(circle_points+1)/circle_points)
        for point in data:
            ax.fill_between(x=point[0] + arc_x, y1=point[1] + arc_y1, y2=point[1] + arc_y2,
                            color=color, linewidth=linewidth, alpha=alpha)
    else:
        circle_x = radius*np.cos(2*np.pi*np.arange(circle_points+1)/circle_points)
        circle_y = radius*np.sin(2*np.pi*np.arange(circle_points+1)/circle_points)
        for point in data:
            ax.plot(point[0] + circle_x, point[1] + circle_y, color=color, linewidth=linewidth, alpha=alpha)




def draw_simplicies(data, edges=[], triangles=[],
                    node_color='indigo', node_width=8, node_alpha=1,
                    edge_color='steelblue', edge_width=4, edge_alpha=1,
                    triangle_color='aquamarine', triangle_alpha=0.7, ax=plt):
    """
    Plots nodes edges and triangles of all simplicies.
    
    Parameters:
    -----------
    data : np.ndarray or list of np.ndarrays
        List of points (nodes).
            
    edges : list of tuples
        List of node pairs.
            
    triangles : list of tuples
        List of triangle triples.
            
    node_color : color or rgba tuple
        Node color.
            
    node_width : scalar
        Linewidth of nodes.
        
    node_alpha : float
        The alpha blending value for node, between 0 (transparent) and 1 (opaque).

    edge_color : color or rgba tuple
        Edge color.
            
    edge_width : scalar
        Linewidth of edges.
        
    edge_alpha : float
        The alpha blending value for edge, between 0 (transparent) and 1 (opaque).
            
    triangle_color : color or rgba tuple
        Triangle fill color.
        
    node_alpha : float
        The alpha blending value for filled triangle, between 0 (transparent) and 1 (opaque).
        
    ax : matplotlib.pyplot or Axes class
    """
    for triangle in triangles:
        xs = [data[i][0] for i in triangle]
        ys = [data[i][1] for i in triangle]
        ax.fill(xs, ys, color=triangle_color, alpha=triangle_alpha)
    for edge in edges:
        p0 = data[edge[0]]
        p1 = data[edge[1]]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=edge_color, linewidth=edge_width, alpha=edge_alpha)
    for point in data:
        ax.scatter(point[0], point[1], color=node_color, linewidth=node_width, alpha=node_alpha)            
        



