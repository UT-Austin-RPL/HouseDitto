import os

import numpy as np
import open3d as o3d
import point_cloud_utils as pcu

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def update_problem_list(problem_list, data, msg='', output_path='problem_list.npz'):
    data['msg'] = msg
    problem_list.append(
        data
    )
    np.savez(output_path, problem_list)
    return problem_list


def check_in_box(xyz_list, obj_bbox_list, offset=0.):
    IS_WITHIN_BBOX = False
    for obj_bbox in obj_bbox_list:                
        pts_idx = obj_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(xyz_list))
        if len(pts_idx) >= 1:
            return False
    return True


def downsample_pc_voxel(pts, sizeof_voxel_pc, color=None, label=None, return_idx=False):
    
    pts_original = pts
    pts_sampled, _, _ = pcu.downsample_point_cloud_voxel_grid(sizeof_voxel_pc, points=pts)
    if len(pts_sampled) >= 10:
        _, pts_idx = pcu.k_nearest_neighbors(pts_sampled, pts, k=1)
    else: # no downsample
        pts_idx = range(len(pts_original))
    
    
    return_list = []
    
    pts = pts[pts_idx]
    return_list.append(pts)
    
    if color is not None:
        assert len(pts_original) == len(color)
        color = color[pts_idx]
        return_list.append(color)
    
    if label is not None:
        assert len(pts_original) == len(label)
        label = label[pts_idx]
        return_list.append(label)
        
    if return_idx:
        return_list.append(pts_idx)
    
    return tuple(return_list)


def visualize_pc(pts, color=None, order='xzy'):    
    fig = plt.figure()
    ax = Axes3D(fig)
    if order == 'xzy':
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=3, c=color)
    elif order == 'xyz':
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=color)
    plt.show()
    plt.close()


def xzy2xyz(arr):
    return np.stack([arr[:, 0], arr[:, 2], arr[:, 1]], axis=1)