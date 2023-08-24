import itertools
import trimesh
import numpy as np
import pybullet as p
import open3d as o3d
from igibson.utils import utils
from igibson.external.pybullet_tools.utils import (
    get_all_links,
    get_joint_info,
)
from utils.utils_transform import get_transform


def sample_bbox_in_wf(obj_list, sampling_method=None, link_list=None, aabb_offset=0.):    
    
    assert sampling_method in (
        'on_link', 'on_obj'
    )
    if sampling_method == 'on_obj':
        assert link_list is None 
    
    bbox_center_list = []
    bbox_orn_list = []
    bbox_bf_extent_list = []
    world_frame_vertex_positions_list = []
        
    for idx_obj, selected_obj in enumerate(obj_list):  
        
        if sampling_method == 'on_link':
            if link_list is None:
                link_list = get_all_links(selected_obj.get_body_id())
                if -1 in link_list:
                    link_list.remove(-1)
        elif sampling_method == 'on_obj':
            link_list = [-1] # when link_id=-1, `get_base_aligned_bounding_box()` returns the bbox of the whole object
        else:
            raise NotImplementedError

        for link in link_list:             
            bbox_center, bbox_orn, bbox_bf_extent, _ = selected_obj.get_base_aligned_bounding_box(link_id=link, xy_aligned=True)
            bbox_center_list.append(bbox_center)
            bbox_orn_list.append(bbox_orn)
            bbox_bf_extent_list.append(bbox_bf_extent) 
            
            bbox_frame_vertex_positions = np.array(list(itertools.product((1, -1), repeat=3))) * (
                bbox_bf_extent / 2 + aabb_offset
            )
            bbox_transform = utils.quat_pos_to_mat(bbox_center, bbox_orn)
            world_frame_vertex_positions = trimesh.transformations.transform_points(
                bbox_frame_vertex_positions, bbox_transform
            )  
            world_frame_vertex_positions_list.append(world_frame_vertex_positions)
    
    result = {
        "bbox_center_list": bbox_center_list, 
        "bbox_orn_list": bbox_orn_list, 
        "bbox_bf_extent_list": bbox_bf_extent_list, 
        "world_frame_vertex_positions_list": world_frame_vertex_positions_list,
    }

    return result   


def get_pt_idx_within_object(pts_opengl, obj, aabb_offset=0.):
    
    # Get obj bbox
    result_on_obj = sample_bbox_in_wf(
        obj_list=[obj],
        sampling_method='on_obj', 
        aabb_offset=aabb_offset,
    )
    # Generate obj bbox via open3d 
    # bbox_list_on_obj = []
    # for world_frame_vertex_positions in result_on_obj["world_frame_vertex_positions_list"]:
    #     bbox = o3d.geometry.AxisAlignedBoundingBox()
    #     bbox = bbox.create_from_points(o3d.utility.Vector3dVector(world_frame_vertex_positions))
    #     bbox_list_on_obj.append(bbox)
    
    # Alternative way
    bbox_list_on_obj = []
    boundaries = p.getAABB(obj.get_body_id())
    bbox = o3d.geometry.AxisAlignedBoundingBox()
    bbox = bbox.create_from_points(o3d.utility.Vector3dVector(boundaries))
    bbox_list_on_obj.append(bbox)
    
    # Sample surface point index within bbox 
    pts_idx_list_on_obj = []
    for bbox in bbox_list_on_obj:
        pts_idx = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pts_opengl))
        pts_idx_list_on_obj.extend(pts_idx)
    
    return pts_idx_list_on_obj


def get_pt_idx_within_object_link(pts_opengl, obj, link_id=None, aabb_offset=0.):
    # Get link bbox
    if link_id is None:
        result_on_link = sample_bbox_in_wf(
            obj_list=[obj],
            sampling_method='on_link', 
            aabb_offset=aabb_offset,
        )
    else:
        result_on_link = sample_bbox_in_wf(
            obj_list=[obj], 
            link_list=[link_id],
            sampling_method='on_link', 
            aabb_offset=aabb_offset,
        )
    
    # Generate link bbox via open3d 
    bbox_list_on_link = []
    for world_frame_vertex_positions in result_on_link["world_frame_vertex_positions_list"]:
        bbox = o3d.geometry.AxisAlignedBoundingBox()
        bbox = bbox.create_from_points(o3d.utility.Vector3dVector(world_frame_vertex_positions))
        bbox_list_on_link.append(bbox)
    
    # Sample surface point index within bbox 
    pts_idx_list_on_link = []
    for bbox in bbox_list_on_link:
        pts_idx = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pts_opengl))
        pts_idx_list_on_link.extend(pts_idx)
    
    return pts_idx_list_on_link


def get_link_bbox_center(obj, link_id):
    result_on_link = sample_bbox_in_wf(
        obj_list=[obj], 
        link_list=[link_id],
        sampling_method='on_link', 
        aabb_offset=0.,
    )
    assert len(result_on_link["world_frame_vertex_positions_list"]) == 1
    
    world_frame_vertex_positions = result_on_link["world_frame_vertex_positions_list"][0]
    link_bbox = o3d.geometry.AxisAlignedBoundingBox()
    link_bbox = link_bbox.create_from_points(o3d.utility.Vector3dVector(world_frame_vertex_positions))
    link_center = np.mean([link_bbox.get_min_bound(), link_bbox.get_max_bound()], axis=0)
    return link_bbox, link_center


def get_joint_screw(obj, link_id):        
    v = p.getJointInfo(
        obj.get_body_id(), 
        link_id
    )
    joint_type = v[2]
    joint_axis = v[-4]
    joint_pos_parent = v[-3]
    joint_ori_parent = v[-2]
    parent_index = v[-1]
    if parent_index == -1: # baselink
        parent_link_state = p.getBasePositionAndOrientation(obj.get_body_id())
    else:
        parent_link_state = p.getLinkState(obj.get_body_id(), parent_index)
    parent_link_trans = get_transform(parent_link_state[0], parent_link_state[1])
    relative_trans = get_transform(joint_pos_parent, joint_ori_parent)
    axis_trans = parent_link_trans * relative_trans
    axis_global = axis_trans.rotation.as_matrix().dot(joint_axis)
    axis_global /= np.sqrt(np.sum(axis_global ** 2))
    point_on_axis = axis_trans.translation
    moment = np.cross(point_on_axis, axis_global)
    
    return axis_global, moment


def get_joint_axis(selected_obj, link):
    # Get the bounding box of the child link.
    (
        bbox_center_in_world,
        bbox_quat_in_world,
        bbox_extent_in_link_frame,
        bbox_center_in_link_frame,
    ) = selected_obj.get_base_aligned_bounding_box(link_id=link, visual=False, link_base=True)

    # Get the part of the object away from the joint position/axis.
    # The link origin is where the joint is. Let's get the position of the origin w.r.t the CoM.
    target_bid = selected_obj.get_body_id()
    dynamics_info = p.getDynamicsInfo(target_bid, link)
    com_wrt_origin = (dynamics_info[3], dynamics_info[4])
    bbox_wrt_origin = p.multiplyTransforms(*com_wrt_origin, bbox_center_in_link_frame, [0, 0, 0, 1])
    origin_wrt_bbox = p.invertTransform(*bbox_wrt_origin)

    joint_axis = np.array(get_joint_info(selected_obj.get_body_id(), link).jointAxis)
    joint_axis /= np.linalg.norm(joint_axis)
    origin_towards_bbox = np.array(bbox_wrt_origin[0])
    open_axis = np.cross(joint_axis, origin_towards_bbox)
    open_axis /= np.linalg.norm(open_axis)
    lateral_axis = np.cross(open_axis, joint_axis)

    # Match the axes to the canonical axes of the link bb.
    lateral_axis_idx = np.argmax(np.abs(lateral_axis))
    open_axis_idx = np.argmax(np.abs(open_axis))
    joint_axis_idx = np.argmax(np.abs(joint_axis))
    assert lateral_axis_idx != open_axis_idx
    assert lateral_axis_idx != joint_axis_idx
    assert open_axis_idx != joint_axis_idx
    
    return joint_axis, lateral_axis, open_axis, joint_axis_idx, lateral_axis_idx, open_axis_idx