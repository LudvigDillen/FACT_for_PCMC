import open3d as o3d
import numpy as np


def vis_pc(pc):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point cloud', width=1900, height=2000)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def vis_2pcs(pc0, pc1):
    pcdl = []
    for i, pc in enumerate([pc0, pc1]):
        zval = np.ones(pc.shape[0])
        cvec = np.zeros((len(zval), 3))
        if i == 0:
            cvec[:, 2] = zval  # All blue colormap
        elif i == 1:
            cvec[:, 1] = zval  # All green colormap
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(cvec)
        pcdl.append(pcd)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pc in pcdl:
        vis.add_geometry(pc)
    vis.run()
    vis.destroy_window()


def vis_2pcs_aligned_vs_misaligned(pc0, pc1_CS1, pc1_CS0, cam_pose=None):
    pcdl = []
    for i, pc in enumerate([pc0, pc1_CS1, pc1_CS0]):
        zval = np.ones(pc.shape[0])
        cvec = np.zeros((len(zval), 3))
        if i == 0:
            cvec[:, 2] = zval  # All blue colormap
        elif i == 1:
            cvec[:, 0] = zval  # All red colormap
        elif i == 2:
            cvec[:, 1] = zval  # All green colormap
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(cvec)
        pcdl.append(pcd)
    pcd_misaligned = [pcdl[0], pcdl[1]]
    pcd_aligned = [pcdl[0], pcdl[2]]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Misaligned point clouds', width=1900, height=2000)
    for pc in pcd_misaligned:
        vis.add_geometry(pc)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='Aligned point clouds', width=1900, height=2000)
    for pc in pcd_aligned:
        vis2.add_geometry(pc)

    while True:
        for pc in pcd_misaligned:
            vis.update_geometry(pc)
        if not vis.poll_events():
            break
        vis.update_renderer()

        for pc in pcd_aligned:
            vis2.update_geometry(pc)

        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis.destroy_window()
    vis2.destroy_window()


# TODO: I do not get think function working for some reason ...
def set_camera_pose(cam_pose, view_control):
    # Get the current camera parameters
    camera_parameters = view_control.convert_to_pinhole_camera_parameters()

    # Update the camera parameters with the new camera pose
    camera_parameters.extrinsic = cam_pose
    # Apply the new camera parameters
    view_control.convert_from_pinhole_camera_parameters(camera_parameters)