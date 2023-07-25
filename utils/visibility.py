import numpy as np
import torch
from scipy.spatial import ConvexHull
import time
# this version was 3 times slower than scipy's version when I compared for the Stanford Bunny
# from pyhull.convex_hull import ConvexHull

from utils.pointclouds import PC


def triangle_angles(A, B, neighbors):
    """Compute the internal angles of triangle ABC in radians."""
    a = np.linalg.norm(B - neighbors, axis=1)
    b = np.linalg.norm(A - neighbors, axis=1)
    c = np.linalg.norm(A - B)

    betas = np.arccos((a**2 + c**2 - b**2) / (2*a*c))
    return betas


def triangle_angles_torch(viewpoint, batch_vertices, batch_neighbors):
    """Compute the internal angles of triangle ABC in radians."""
    dtype = batch_vertices.dtype

    batch_vertices_reshaped = batch_vertices.unsqueeze(1)  # Add an extra dimension for broadcasting
    viewpoint_reshaped = viewpoint.unsqueeze(0).unsqueeze(0)  # Add an extra dimension for broadcasting

    a = torch.norm(batch_vertices_reshaped - batch_neighbors, dim=2)
    b = torch.norm(viewpoint_reshaped - batch_neighbors, dim=2)
    c = torch.norm(viewpoint_reshaped - batch_vertices_reshaped, dim=2)

    betas = torch.acos((a**2 + c**2 - b**2) / (2*a*c))

    # Create a mask for valid neighbors
    mask = (batch_neighbors != 0).any(dim=2).to(dtype)

    # Apply the mask to the computed angles
    betas_valid = betas*mask
    return betas_valid


# def calculate_visibility_angles(viewpoint, pc, vertices_map, input_convex_hull, batch_size):
#     t1 = time.time()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Make sure your data is on the GPU
#     pc = torch.tensor(pc, device=device)
#     dtype = pc.dtype
#     viewpoint = torch.tensor(viewpoint, device=device, dtype=dtype)
#     input_convex_hull = torch.tensor(input_convex_hull, device=device, dtype=dtype)

#     visibility_angles = torch.zeros(len(vertices_map), device=device, dtype=dtype)
#     keys = list(vertices_map.keys())
#     t2 = time.time()
#     acc1 = 0
#     acc2 = 0
#     for i in range(0, len(keys), batch_size):
#         t11 = time.time()
#         batch_keys = keys[i:i+batch_size]

#         # Determine N_max_neighbors for this batch
#         N_max_neighbors = max(len(vertices_map[k]) for k in batch_keys)

#         # Create a tensor initialized with zeros
#         batch_neighbors = torch.zeros(len(batch_keys), N_max_neighbors, 3, device=device, dtype=dtype)

#         # Fill in the neighbors for each vertex in the batch
#         for j, k in enumerate(batch_keys):
#             n_neighbors = len(vertices_map[k])
#             batch_neighbors[j, :n_neighbors] = input_convex_hull[vertices_map[k]]
#         t22 = time.time()
#         acc1 += t22-t11

#         # Compute angles
#         angles = triangle_angles_torch(viewpoint, pc[batch_keys], batch_neighbors)
#         visibility_angles[i:i+batch_size] = torch.sum(angles, dim=1)
#         t33 = time.time()
#         acc2 += t33-t22
#     print(f"Timings\n {np.around(t2-t1, 3)}, {np.around(acc1, 3)}, {np.around(acc2, 3)}")
#     return visibility_angles.cpu().numpy()


def calculate_visibility_angles(viewpoint, pc, vertices_map, input_convex_hull, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pc.dtype
    # Convert data to PyTorch tensors on the CPU
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

        # Create a tensor initialized with zeros
        batch_neighbors = torch.zeros(N_batch_keys, N_max_neighbors, 3, device=device, dtype=dtype)

        padded_indices = torch.zeros(N_batch_keys, N_max_neighbors, dtype=torch.long)

        for j, ind in enumerate(neighbor_indices):
            padded_indices[j, :neighbor_lengths[j]] = torch.tensor(ind, dtype=torch.long)

        mask = (padded_indices != 0).unsqueeze(dim=2).to(device)
        batch_neighbors[:, :N_max_neighbors] = mask*input_convex_hull[padded_indices]
        # Compute angles
        angles = triangle_angles_torch(viewpoint, pc[batch_keys], batch_neighbors)
        # angles_masked = angles.clone()
        # angles_masked[~(mask.squeeze(dim=2))] = float('inf')
        # two_smallest_angles = torch.topk(angles_masked, k=2, dim=1, largest=False).values
        # two_smallest_angles[two_smallest_angles == float('inf')] = torch.pi/2
        # visibility_angles[i:i+batch_size] = torch.sum(angles, dim=1)
        visibility_angles[i:i+batch_size] = torch.sum(angles, dim=1)/torch.tensor(neighbor_lengths,
                                                                                  dtype=dtype, device=device)
    return visibility_angles


def visible_points(pc: np.array, viewpoint: np.array, hpr_radius: float, compute_weights=False,
                   inversion_kernel="spherical flipping") -> np.array:
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

    # TODO: Make this general, always take tensors as input, and remove this then ...

    # Move all points such that the viewpoint is in the origin
    pc_vp = pc - viewpoint

    # Calculate distance from origin to all points
    pc_dist = torch.norm(pc_vp, dim=1)

    # SPHERICAL FLIPPING
    if inversion_kernel == "spherical flipping":
        # Choosing dynamic radius
        R = 10**(hpr_radius)*torch.max(pc_dist)
        # Perform spherical flipping
        flipped_pc_vp = pc_vp + 2*((R-pc_dist)/pc_dist)[:, None]*pc_vp

    # EXPONENTIAL INVERSION KERNEL
    if inversion_kernel == "exponential inversion kernel":
        # Move all points such that the viewpoint is in the origin
        pc_vp = pc - viewpoint
        # Calculate distance from origin to all points
        pc_dist = torch.norm(pc_vp, dim=1)
        # Exponential inversion, TODO: Set parameter 0.0001 to something good ...
        flipped_pc_vp = pc_vp*(pc_dist**(-0.0001)/pc_dist)[:, None]

    # Add viewpoint to the set
    input_convex_hull = torch.cat((flipped_pc_vp, viewpoint[None, :]), dim=0)

    # Compute the convex hull
    n_points = flipped_pc_vp.shape[0]
    if compute_weights:
        ch = ConvexHull(input_convex_hull.cpu())
        visible_inds = ch.vertices
    else:
        visible_inds = ConvexHull(input_convex_hull.cpu()).vertices

        # Remove the potential viewpoint from the inds list
        if n_points in visible_inds:
            visible_inds = visible_inds[0:-1]
        return visible_inds

    # New code in progress ...
    # TODO: Do this in batches, do it better ...
    # Dictionary to store angles for each vertex
    vertices_map = {i: [] for i in visible_inds}
    # This code might not look pretty but is quite fast at least
    # (having an inner loop seems to make it slower)
    for vertices in ch.simplices:
        vertex = vertices[0]
        if vertices[1] not in vertices_map[vertex]:
            vertices_map[vertex].append(vertices[1])
        if vertices[2] not in vertices_map[vertex]:
            vertices_map[vertex].append(vertices[2])

        vertex = vertices[1]
        if vertices[0] not in vertices_map[vertex]:
            vertices_map[vertex].append(vertices[0])
        if vertices[2] not in vertices_map[vertex]:
            vertices_map[vertex].append(vertices[2])

        vertex = vertices[2]
        if vertices[0] not in vertices_map[vertex]:
            vertices_map[vertex].append(vertices[0])
        if vertices[1] not in vertices_map[vertex]:
            vertices_map[vertex].append(vertices[1])

    # Remove the potential viewpoint from the inds list
    if n_points in vertices_map:
        del vertices_map[n_points]

    # TODO: Set batch size here to what we give in the cls-file
    visibility_angles = calculate_visibility_angles(
        viewpoint, pc, vertices_map, input_convex_hull, batch_size=256)
    # visibility score (inspired by "On the Visibility of Point Clouds")
    visibility_scores = visibility_angles
    # New code in progress ends

    # Remove the potential viewpoint from the inds list
    if n_points in visible_inds:
        visible_inds = visible_inds[0:-1]
    return visible_inds, visibility_scores


def covisible_inds(pc0, pc_union, T0, T1, hpr_radius=3.25):
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

    vis_points_pose0 = visible_points(pc_union, zero_vec, hpr_radius)
    # init with zeros (regard all point as not visible)
    visible_mask0 = torch.zeros(union_n_points, device=device, dtype=torch.int)
    # Set the visible points to 1
    visible_mask0[vis_points_pose0] = 1
    # All points from PC0 should be marked as visible
    visible_mask0[ind_list < pc0_n_points] = 1

    viewpoint1_CS0 = torch.matmul(torch.inverse(T0), T1)[:3, 3]
    vis_points_pose1 = visible_points(pc_union, viewpoint1_CS0, hpr_radius)
    # init with zeros (regard all point as not visible)
    visible_mask1 = torch.zeros(union_n_points, device=device, dtype=torch.int)
    # Set the visible points to 1
    visible_mask1[vis_points_pose1] = 1
    # All points from PC1 should be marked as visible
    visible_mask1[ind_list >= pc0_n_points] = 1

    visible_mask = visible_mask0*visible_mask1
    visible_inds = ind_list[visible_mask != 0]
    visible_inds_pc0 = visible_inds[visible_inds < pc0_n_points]
    visible_inds_pc1 = visible_inds[visible_inds >= pc0_n_points] - pc0_n_points
    return visible_inds, visible_inds_pc0, visible_inds_pc1


def keep_covisible_points(PC0, PC1, PC_union, T0, T1, hpr_radius=3.25):
    visible_inds, visible_inds_pc0, visible_inds_pc1 = covisible_inds(
        PC0.pc, PC_union.pc, T0, T1, hpr_radius=hpr_radius)
    device = PC0.device
    PC_union_covisible = PC(PC_union.pc[visible_inds], PC_union.distances_to_origin[visible_inds], label=2,
                            device=device)
    PC0_covisible = PC(PC0.pc[visible_inds_pc0], PC0.distances_to_origin[visible_inds_pc0], label=0,
                       device=device)
    PC1_covisible = PC(PC1.pc[visible_inds_pc1], PC1.distances_to_origin[visible_inds_pc1], label=1,
                       device=device)
    return PC0_covisible, PC1_covisible, PC_union_covisible


    # For each vertex in the convex hulls
    # visibility_angles = np.zeros(visible_inds.shape[0])
    # for i, vertex_key in enumerate(vertices_map):
    #     # For each neighboring vertex
    #     neighbors = input_convex_hull[vertices_map[vertex_key]]
    #     # We do not care about the viewpoint vertex
    #     if vertex_key == pc.shape[0]:
    #         continue
    #     vertex = pc[vertex_key]

    #     # Calculate the angles of the triangle formed by the vertex, its neighbor, and the viewpoint
    #     angles = triangle_angles(viewpoint, vertex, neighbors)

    #     # Get the angle at the current vertex
    #     visibility_angles[i] = np.sum(angles)