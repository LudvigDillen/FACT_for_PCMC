import numpy as np
from scipy.spatial import ConvexHull
# this version was 3 times slower than scipy's version when I compared for the Stanford Bunny
#from pyhull.convex_hull import ConvexHull


def visible_points(pc: np.array, viewpoint: np.array, param_radius: float) -> np.array:
    """
    Returns the indices of points in a given point cloud that are visible from a specified viewpoint. 

    The function implements the method presented in the article "Direct Visibility of Point Sets".

    Parameters
    ----------
    pc : np.array
        A point cloud represented as an (n, 3) array, where 'n' is the number of points in the cloud,
        and each point is a 3D coordinate in the format (x, y, z).

    viewpoint : np.array
        A array representing the viewpoint from which visibility is assessed. This is a single 3D coordinate
        in the format (x, y, z).

    param_radius: float
        An parameter affecting the radius used in the spherical flipping. A smaller value may cause more 
        visible points to labeled as non-visible while a larger value may cause non-visible points to be
        labeled as visible. See more info in the referenced paper on this.

    Returns
    -------
    np.array
        A array containing the indices of the points in the input point cloud that are visible
        from the given viewpoint.

    Notes
    -----
    Visibility of points is determined according to the method presented in "Direct Visibility of Point Sets". 
    """
    # Move all points such that the viewpoint is in the origin
    pc_vp = pc - viewpoint
    # Calculate distance from origin to all points
    pc_dist = np.linalg.norm(pc_vp, axis=1)
    # Choosing dynamic radius
    R = 10**(param_radius)*np.max(pc_dist)
    # Perform spherical flipping
    flipped_pc_vp = pc_vp + 2*((R-pc_dist)/pc_dist)[:, np.newaxis]*pc_vp
    # Compute the convex hull
    visible_inds = ConvexHull(flipped_pc_vp).vertices
    return visible_inds
