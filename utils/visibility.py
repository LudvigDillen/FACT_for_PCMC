#import torch
import numpy as np


def visible_points(point_cloud: np.array, viewpoint: np.array) -> np.array:
    """
    Returns the indices of points in a given point cloud that are visible from a specified viewpoint. 

    The function implements the method presented in the article "Direct Visibility of Point Sets".

    Parameters
    ----------
    point_cloud : np.array
        A point cloud represented as an (n, 3) array, where 'n' is the number of points in the cloud,
        and each point is a 3D coordinate in the format (x, y, z).
    
    viewpoint : np.array
        A array representing the viewpoint from which visibility is assessed. This is a single 3D coordinate
        in the format (x, y, z).

    Returns
    -------
    np.array
        A array containing the indices of the points in the input point cloud that are visible 
        from the given viewpoint. 

    Notes
    -----
    Visibility of points is determined according to the method presented in "Direct Visibility of Point Sets". 
    """
    n_points, dim = point_cloud.shape
    # Move point to the 
    print(n_points)
    return visible_points