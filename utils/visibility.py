import torch
from scipy.spatial import ConvexHull
import time
# this version was 3 times slower than scipy's version when I compared for the Stanford Bunny
# from pyhull.convex_hull import ConvexHull

from utils.pointclouds import PC


def triangle_angles(viewpoint, batch_vertices, batch_neighbors, mask):
    """Compute the internal angles of triangle ABC in radians."""
    batch_vertices_reshaped = batch_vertices.unsqueeze(1)  # Add an extra dimension for broadcasting
    viewpoint_reshaped = viewpoint.view(1, 1, -1)  # Add extra dimensions for broadcasting

    a = torch.norm(batch_vertices_reshaped - batch_neighbors, dim=2)
    b = torch.norm(viewpoint_reshaped - batch_neighbors, dim=2)
    c = torch.norm(viewpoint_reshaped - batch_vertices_reshaped, dim=2)

    a_square = torch.pow(a, 2)
    b_square = torch.pow(b, 2)
    c_square = torch.pow(c, 2)

    arg_beta = (a_square + c_square - b_square) / (2*a*c)
    arg_gamma = (a_square + b_square - c_square) / (2*a*b)
    # Theoretically, this is not necessary, but numerically the argument could slightly leave the range
    # resulting in NaN values.
    arg_beta_clipped = torch.clamp(arg_beta, -1, 1)
    betas = torch.acos(arg_beta_clipped)

    arg_gamma_clipped = torch.clamp(arg_gamma, -1, 1)
    gammas = torch.acos(arg_gamma_clipped)

    # Apply the mask to the computed angles
    gammas[torch.isnan(gammas)] = 0
    betas[torch.isnan(betas)] = 0

    angle_diff = mask*(gammas-betas)
    return angle_diff


def calculate_visibility_angles(viewpoint, pc_inverted, vertices_map, batch_size):
    device = pc_inverted.device
    dtype = pc_inverted.dtype

    visibility_angles = torch.zeros(len(vertices_map), device=device, dtype=dtype)
    keys = list(vertices_map.keys())
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i+batch_size]
        N_batch_keys = len(batch_keys)

        # Determine N_max_neighbors for this batch
        neighbor_indices = []
        neighbor_lengths = []
        for k in batch_keys:
            elements = vertices_map[k]
            neighbor_indices.append(elements)
            neighbor_lengths.append(len(elements))
        N_max_neighbors = max(neighbor_lengths)

        padded_indices = torch.zeros(N_batch_keys, N_max_neighbors, dtype=torch.long)

        for j, inds in enumerate(neighbor_indices):
            padded_indices[j, :neighbor_lengths[j]] = torch.tensor(inds, dtype=torch.long)

        mask = (padded_indices != 0).to(device)
        batch_neighbors = (mask.unsqueeze(dim=2))*pc_inverted[padded_indices]

        angles = triangle_angles(viewpoint, pc_inverted[batch_keys], batch_neighbors, mask)

        # Whether to use the sum of the two smallest angles, take the mean (like below), or sum, is somewhat
        # unclear from the article "On the Visibility of Point Clouds"
        # But to me, it makes the most sense to take the mean, since in 3D, a point can have unlimited
        # number of neighbors so summing is definitely not appropriate.
        # Selecting the two smallest ones feels somewhat unmotivated, and might favor points with many
        # neighbors, which does not necessarily simply that those points are visible.

        # However, taking the sum seems to work best is also the recommended way in the article. So,
        # I'll stick with that going forward.

        # MEAN
        visibility_angles[i:i+batch_size] = torch.sum(angles, dim=1)/torch.tensor(neighbor_lengths,
                                                                                  dtype=dtype, device=device)
    return visibility_angles


def visible_points(pc, viewpoint, hpr_radius, gamma=-0.0001, inversion_kernel="exponential",
                   compute_weights=False, batch_size=256):
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

    hpr_radius: float
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
    Visibility Scores is based on the method presented in "On the Visibility of Point Clouds"
    """
    # There are two drawbacks with this function.
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

    # Move all points such that the viewpoint is in the origin
    pc_vp = pc - viewpoint
    del pc  # We should only use pc_vp ...
    viewpoint_origin = viewpoint - viewpoint
    del viewpoint

    # Calculate distance from origin to all points
    pc_dist = torch.norm(pc_vp, dim=1)

    # SPHERICAL FLIPPING
    if inversion_kernel == "spherical_flipping":
        # Choosing dynamic radius
        R = 10**(hpr_radius)*torch.max(pc_dist)
        # Perform spherical flipping
        pc_inverted = pc_vp + 2*((R-pc_dist)/pc_dist)[:, None]*pc_vp

    # EXPONENTIAL INVERSION KERNEL
    if inversion_kernel == "exponential":
        # Exponential inversion, TODO: Set parameter gamma to something good ...
        pc_inverted = pc_vp*(pc_dist**(gamma-1))[:, None]

    # Add viewpoint to the set
    input_convex_hull = torch.cat((pc_inverted, viewpoint_origin[None, :]), dim=0)

    # Compute the convex hull
    n_points = pc_inverted.shape[0]
    if compute_weights:
        ch = ConvexHull(input_convex_hull.cpu())
        visible_inds = ch.vertices
    else:
        visible_inds = ConvexHull(input_convex_hull.cpu()).vertices

        # Remove the potential viewpoint from the inds list
        if n_points in visible_inds:
            visible_inds = visible_inds[0:-1]
        return visible_inds

    # Dictionary to store angles for each vertex
    vertices_map = {i: [] for i in visible_inds}
    # This code might not look pretty but is quite fast at least
    # (having an inner loop seems to make it slower)
    for vertices in ch.simplices:
        insert_vertex = vertices[0]
        if insert_vertex != n_points:
            vertex_key = vertices[1]
            if insert_vertex not in vertices_map[vertex_key]:
                vertices_map[vertex_key].append(insert_vertex)
            vertex_key = vertices[2]
            if insert_vertex not in vertices_map[vertex_key]:
                vertices_map[vertex_key].append(insert_vertex)

        insert_vertex = vertices[1]
        if insert_vertex != n_points:
            vertex_key = vertices[0]
            if insert_vertex not in vertices_map[vertex_key]:
                vertices_map[vertex_key].append(insert_vertex)
            vertex_key = vertices[2]
            if insert_vertex not in vertices_map[vertex_key]:
                vertices_map[vertex_key].append(insert_vertex)

        insert_vertex = vertices[2]
        if insert_vertex != n_points:
            vertex_key = vertices[0]
            if insert_vertex not in vertices_map[vertex_key]:
                vertices_map[vertex_key].append(insert_vertex)
            vertex_key = vertices[1]
            if insert_vertex not in vertices_map[vertex_key]:
                vertices_map[vertex_key].append(insert_vertex)

    # Remove the potential viewpoint from the inds list
    if n_points in vertices_map:
        del vertices_map[n_points]
        visible_inds = visible_inds[0:-1]

    visibility_angles = calculate_visibility_angles(
        viewpoint_origin, pc_inverted, vertices_map, batch_size=batch_size)

    # TODO: Do I want this normalization ... ?
    visibility_scores = (visibility_angles - visibility_angles.min())
    visibility_scores = visibility_scores/visibility_scores.max()
    return visible_inds, visibility_scores


def covisible_inds(pc0, pc_union, T0, T1, hpr_radius=3.25, gamma=-0.0001,
                   inversion_kernel="exponential", compute_weights=False,
                   batch_size=256):
    # We assume here that no non-covisible point will be become visible when adding
    # additional points to the point cloud. This will not necessarily be true in
    # practice but it should hold in theory. Hence, we only need to study the
    # joint point clouds
    device = pc0.device
    dtype = pc0.dtype

    pc0_n_points = pc0.shape[0]
    union_n_points = pc_union.shape[0]

    zero_vec = torch.zeros((3), device=device, dtype=dtype)
    ind_list = torch.arange(0, union_n_points, device=device)

    if compute_weights:
        vis_points_pose0, visibility_scores0 = visible_points(
            pc_union, zero_vec, hpr_radius, gamma=gamma, inversion_kernel=inversion_kernel,
            compute_weights=compute_weights, batch_size=batch_size)
    else:
        vis_points_pose0 = visible_points(
            pc_union, zero_vec, hpr_radius, gamma=gamma, inversion_kernel=inversion_kernel,
            compute_weights=compute_weights, batch_size=batch_size)

    # init with zeros (regard all points as not visible)
    visible_mask0 = torch.zeros(union_n_points, device=device, dtype=torch.int)
    # Set the visible points to 1
    visible_mask0[vis_points_pose0] = 1
    # All points from PC0 should be marked as visible
    visible_mask0[ind_list < pc0_n_points] = 1

    viewpoint1_CS0 = torch.matmul(torch.inverse(T0), T1)[:3, 3]

    if compute_weights:
        vis_points_pose1, visibility_scores1 = visible_points(
            pc_union, viewpoint1_CS0, hpr_radius, gamma=gamma, inversion_kernel=inversion_kernel,
            compute_weights=compute_weights, batch_size=batch_size)
    else:
        vis_points_pose1 = visible_points(
            pc_union, viewpoint1_CS0, hpr_radius, gamma=gamma, inversion_kernel=inversion_kernel,
            compute_weights=compute_weights, batch_size=batch_size)

    # init with zeros (regard all points as not visible)
    visible_mask1 = torch.zeros(union_n_points, device=device, dtype=torch.int)
    # Set the visible points to 1
    visible_mask1[vis_points_pose1] = 1
    # All points from PC1 should be marked as visible
    visible_mask1[ind_list >= pc0_n_points] = 1

    visible_mask = visible_mask0*visible_mask1
    visible_inds = ind_list[visible_mask != 0]
    visible_inds_pc0 = visible_inds[visible_inds < pc0_n_points]
    visible_inds_pc1 = visible_inds[visible_inds >= pc0_n_points] - pc0_n_points

    if compute_weights is False:
        return visible_inds, visible_inds_pc0, visible_inds_pc1

    # init with zeros (regard all points as not visible)
    visibility_scores_mask0 = torch.zeros(union_n_points, device=device, dtype=dtype)
    # Set the estimated visibility scores
    visibility_scores_mask0[vis_points_pose0] = visibility_scores0
    # All points from PC0 should be marked as fully visible
    visibility_scores_mask0[ind_list < pc0_n_points] = 1

    # init with zeros (regard all points as not visible)
    visibility_scores_mask1 = torch.zeros(union_n_points, device=device, dtype=dtype)
    # Set the estimated visibility scores
    visibility_scores_mask1[vis_points_pose1] = visibility_scores1
    # All points from PC0 should be marked as fully visible
    visibility_scores_mask1[ind_list >= pc0_n_points] = 1

    covisibility_scores = visibility_scores_mask0*visibility_scores_mask1
    covisibility_scores = covisibility_scores[visible_mask != 0]

    return visible_inds, visible_inds_pc0, visible_inds_pc1, covisibility_scores


def keep_covisible_points(PC0, PC1, PC_union, T0, T1, compute_weights, hpr_radius=3.25, gamma=-0.0001,
                          inversion_kernel="exponential", batch_size=256):
    if compute_weights:
        visible_inds, visible_inds_pc0, visible_inds_pc1, covisibility_scores = covisible_inds(
            PC0.pc, PC_union.pc, T0, T1, gamma=gamma, inversion_kernel=inversion_kernel,
            compute_weights=compute_weights, hpr_radius=hpr_radius, batch_size=batch_size)
    else:
        visible_inds, visible_inds_pc0, visible_inds_pc1 = covisible_inds(
            PC0.pc, PC_union.pc, T0, T1, gamma=gamma, inversion_kernel=inversion_kernel,
            compute_weights=compute_weights, hpr_radius=hpr_radius, batch_size=batch_size)

    device = PC0.device
    PC_union_covisible = PC(PC_union.pc[visible_inds], PC_union.distances_to_origin[visible_inds], label=2,
                            device=device)
    if compute_weights:
        PC_union_covisible.set_covisibility_weight(covisibility_scores)

    PC0_covisible = PC(PC0.pc[visible_inds_pc0], PC0.distances_to_origin[visible_inds_pc0], label=0,
                       device=device)
    PC1_covisible = PC(PC1.pc[visible_inds_pc1], PC1.distances_to_origin[visible_inds_pc1], label=1,
                       device=device)
    return PC0_covisible, PC1_covisible, PC_union_covisible
