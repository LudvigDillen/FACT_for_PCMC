import numpy as np
import torch
from scipy.spatial import ConvexHull
# this version was 3 times slower than scipy's version when I compared for the Stanford Bunny
# from pyhull.convex_hull import ConvexHull

from utils.pointclouds import PC


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
    # TODO: There are two drawbacks with this function.
    # 1. It does not take the orientation into account. This is necessary to model the points that
    #    are in the field of view.
    # 2. Even, with not changing the orientation, due to translational movements the points will
    #    not be covisible due to the limited field of view.
    # Explanation:
    # In the horizontal direction the field of view is 360 degrees (a complete revolution)
    # but in the vertical direction, the field of view is (+10 deg. to -30.67 deg) so 41.33 deg,
    # and we would need 180 (probably not necessary with so much) to cover the complete
    # surrounding environment. Since the agent is moving, the covisibility will not only change due
    # to occlusions e.g. but also due to orientation which since objects that were in the
    # vertical field of view might not be in the vertical field of view anymore since we might
    # move closer to the object.

    if torch.is_tensor(pc):
        pc = pc.cpu().numpy()
    if torch.is_tensor(viewpoint):
        viewpoint = viewpoint.cpu().numpy()
    # Move all points such that the viewpoint is in the origin
    pc_vp = pc - viewpoint
    # Calculate distance from origin to all points
    pc_dist = np.linalg.norm(pc_vp, axis=1)
    # Choosing dynamic radius
    R = 10**(param_radius)*np.max(pc_dist)
    # Perform spherical flipping
    flipped_pc_vp = pc_vp + 2*((R-pc_dist)/pc_dist)[:, np.newaxis]*pc_vp
    # Add viewpoint to the set
    input_convex_hull = np.append(flipped_pc_vp, viewpoint[np.newaxis, :], axis=0)
    # Compute the convex hull
    visible_inds = ConvexHull(input_convex_hull).vertices
    # Remove the potential viewpoint from the inds list
    n_points = input_convex_hull.shape[0]
    if n_points in visible_inds:
        visible_inds = visible_inds[0:-1]
    return visible_inds


# def keep_covisible_points(PC0, PC1, PC_union, T0, T1):
#     param_radius = 3.7
#     zero_vec = np.zeros((3))

#     # Start with some checks, remove this code later
#     vis_points = visible_points(PC0.pc, zero_vec, param_radius)
#     N_cov_points = len(vis_points)
#     print(f"visible points {N_cov_points} of total {PC0.N_points}")
#     vis_points = visible_points(PC1.pc, zero_vec, param_radius)
#     N_cov_points = len(vis_points)
#     print(f"visible points {N_cov_points} of total {PC1.N_points}")

#     # Now, check the covisible points between the two point clouds
#     viewpoint1_CS0 = torch.matmul(torch.linalg.inv(T0), T1)
#     t_vec = viewpoint1_CS0[:3, 3]
#     vis_points = visible_points(PC0.pc, t_vec, param_radius)
#     N_cov_points = len(vis_points)
#     print(f"visible points {N_cov_points} of total {PC0.N_points}")

#     viewpoint0_CS1 = torch.matmul(torch.linalg.inv(T1), T0)
#     t_vec = viewpoint0_CS1[:3, 3]
#     vis_points = visible_points(PC1.pc, t_vec, param_radius)
#     N_cov_points = len(vis_points)
#     print(f"visible points {N_cov_points} of total {PC1.N_points}")

#     # Do actual implementation ... (below ...)
#     visible_mask = np.array(0, PC_union.N_points)
#     vis_points_pose0 = visible_points(PC_union.pc, zero_vec, param_radius)
#     N_vis_points_pose0 = len(vis_points_pose0)
#     print(f"visible points {N_vis_points_pose0} of total {PC_union.N_points}")
#     vis_points_pose1 = visible_points(PC_union.pc, viewpoint1_CS0, param_radius)
#     N_vis_points_pose1 = len(vis_points_pose1)
#     print(f"visible points {N_vis_points_pose1} of total {PC_union.N_points}")
#     PC_pair_covisible = None
#     return PC_pair_covisible, None, None

def keep_covisible_points(PC0, PC1, PC_union, T0, T1):
    # We assume here that no non-covisible point will be become visible when adding
    # additional points to the point cloud. This will not necessarily be true in
    # practice but it should hold in theory. Hence, we only need to study the 
    # joint point cloud.

    param_radius = 3.7
    zero_vec = np.zeros((3))
    ind_list = np.arange(0, PC_union.N_points)

    vis_points_pose0 = visible_points(PC_union.pc, zero_vec, param_radius)
    # init with zeros (regard all point as not visible)
    visible_mask0 = np.zeros(PC_union.N_points)
    # Set the visible points to 1
    visible_mask0[vis_points_pose0] = 1
    # All points from PC0 should be marked as visible
    visible_mask0[ind_list < PC0.N_points] = 1

    viewpoint0_CS1 = torch.matmul(torch.linalg.inv(T1), T0)[:3, 3]
    vis_points_pose1 = visible_points(PC_union.pc, viewpoint0_CS1, param_radius)
    # init with zeros (regard all point as not visible)
    visible_mask1 = np.zeros(PC_union.N_points)
    # Set the visible points to 1
    visible_mask1[vis_points_pose1] = 1
    # All points from PC1 should be marked as visible
    visible_mask1[ind_list >= PC0.N_points] = 1

    visible_mask = visible_mask0*visible_mask1
    visible_inds = ind_list[visible_mask != 0]
    visible_inds_pc0 = visible_inds[visible_inds < PC0.N_points]
    visible_inds_pc1 = visible_inds[visible_inds >= PC0.N_points] - PC0.N_points

    PC_union_covisible = PC(PC_union.pc[visible_inds], PC_union.distances_to_origin[visible_inds])
    PC0_covisible = PC(PC0.pc[visible_inds_pc0], PC0.distances_to_origin[visible_inds_pc0])
    PC1_covisible = PC(PC1.pc[visible_inds_pc1], PC1.distances_to_origin[visible_inds_pc1])
    return PC0_covisible, PC1_covisible, PC_union_covisible
