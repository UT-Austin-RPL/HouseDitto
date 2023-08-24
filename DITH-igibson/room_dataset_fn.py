import os
import numpy as np
import pybullet as p

import igibson
from igibson.external.pybullet_tools.utils import (
    get_joint_type,
)    
from igibson.utils.utils import restoreState
from utils.utils_object import get_joint_screw, get_pt_idx_within_object_link
from sklearn.metrics.pairwise import euclidean_distances


from igibson.utils.assets_utils import get_cubicasa_scene_path


def update_problem_list(problem_list, data, msg='', output_path='problem_list.npz'):
    data['msg'] = msg
    problem_list.append(
        data
    )
    np.savez(output_path, problem_list)
    return problem_list


def filter_cubicasa_scene_list(scene_dir, save_dir=None, scene_failed_dir=None):

    scene_dir = os.path.join(igibson.cubicasa_dataset_path, 'scenes')
    scene_list = [o for o in os.listdir(scene_dir) 
                    if os.path.isdir(os.path.join(scene_dir,o))]
    scene_list.sort(reverse=False)
    
    scene_removed_list = []
    if save_dir is not None:
        scene_existed_list = [o for o in os.listdir(save_dir) 
                            if not os.path.isdir(os.path.join(save_dir,o))]
        scene_removed_list.extend(scene_existed_list)
    if scene_failed_dir is not None:
        scene_failed_list = [o for o in os.listdir(scene_failed_dir) 
                            if os.path.isdir(os.path.join(scene_failed_dir, o))]
        scene_removed_list.extend(scene_failed_list)
    
    
    scene_list_filtered = []
    for scene_name in scene_list:
        SHOULD_BE_REMOVED = False
        
        # check if in removed list
        for scene_name_removed in scene_removed_list:
            if scene_name_removed.startswith(scene_name):
                SHOULD_BE_REMOVED = True
                break
            
        # check other factors
        if  (( not SHOULD_BE_REMOVED
            ) and 
            os.path.exists(
                get_cubicasa_scene_path(scene_name)
            ) and 
            os.path.exists(
                os.path.join(get_cubicasa_scene_path(scene_name), 'urdf', '%s_best.urdf' % scene_name)
            ) and 
            os.path.exists(
                os.path.join(get_cubicasa_scene_path(scene_name), 'layout', 'floor_trav_0.png')
            ) and
            os.path.exists(
                os.path.join(get_cubicasa_scene_path(scene_name), 'layout', 'floor_trav_no_obj_0.png')
            )):
                scene_list_filtered.append(scene_name)
    
    print('scene_list:', len(scene_list))
    print('scene_list_filtered:', len(scene_list_filtered))
    
    return scene_list_filtered


def batch_perpendicular_line(
    x: np.ndarray, l: np.ndarray, pivot: np.ndarray
) -> np.ndarray:
    """
    x: B * 3
    l: 3
    pivot: 3
    p_l: B * 3
    """
    offset = x - pivot
    p_l = offset.dot(l)[:, np.newaxis] * l[np.newaxis] - offset
    dist = np.sqrt(np.sum(p_l ** 2, axis=-1))
    p_l = p_l / (dist[:, np.newaxis] + 1.0e-5)
    return p_l, dist


def check_collision_for_gripper(
    pts_on_link, 
    #
    simulator,
    selected_obj, 
    link, 
    constraint_marker, 
    pb_initial_state, 
    #
    pts_idx_list_on_link,
    #
    collision_offset=0.03,
    aabb_offset=-0.1,
):
    pos_pt_list = []
    neg_pt_list = []
    pos_pt_idx_list = []
    neg_pt_idx_list = []
    
    joint_type = get_joint_type(selected_obj.get_body_id(), link)
    assert joint_type in [0, 1]
    
    screw_axis, screw_moment = get_joint_screw(selected_obj, link)
    screw_axis_idx = np.argmax(np.abs(screw_axis))
    joint_axis_idx = screw_axis_idx
    
    # get joint info
    if joint_type == 0:        
        screw_point = np.cross(screw_axis, screw_moment)
        # find by maximum distance between pts and axis line
        p2l_vec, p2l_dist = batch_perpendicular_line(
            pts_on_link, screw_axis, screw_point
        )
        idx = p2l_dist.argmax()
        farthest_point = pts_on_link[idx]
        lateral_axis = (farthest_point - screw_point)
        lateral_axis_2 = lateral_axis.copy()
        lateral_axis_2[screw_axis_idx] = 0.
        lateral_axis_idx = np.argmax(np.abs(lateral_axis_2))
    
    
    for idx_pt_on_link, pt in enumerate(pts_on_link):
        
        if get_joint_type(selected_obj.get_body_id(), link) == 0: # revolute
            # check two positions
            ENOUGH_SPACE_A = False
            ENOUGH_SPACE_B = False
            
            # first position
            constraint_marker.set_position([0., 0., 100.])
            restoreState(pb_initial_state)
            test_position = pt.copy()
            test_position[lateral_axis_idx] += collision_offset
            constraint_marker.set_position(test_position)                    
            simulator.step()
            
            if len(get_pt_idx_within_object_link(test_position.reshape(1, 3), selected_obj, link, aabb_offset=aabb_offset)) == 0:
                body1 = constraint_marker.get_body_id()
                body2 = selected_obj.get_body_id()
                max_distance = 0    
                collision = len(
                    p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=-1, linkIndexB=link, distance=max_distance)
                ) != 0  # getContactPoints 
                
                screw_axis, screw_moment = get_joint_screw(selected_obj, link)
                point_on_axis = np.cross(screw_axis, screw_moment)
                pt_copy = pt.copy()
                pt_copy[joint_axis_idx] = 0. # ignore z-axis
                point_on_axis[joint_axis_idx] = 0. # ignore z-axis
                distance = euclidean_distances(pt_copy.reshape(1, -1), point_on_axis.reshape(1, -1)) 
                too_close_to_axis = distance < 0.1
                
                if not collision and not too_close_to_axis:
                    ENOUGH_SPACE_A = True
            
            
            # second position
            constraint_marker.set_position([0., 0., 100.])
            restoreState(pb_initial_state)
            test_position = pt.copy()
            test_position[lateral_axis_idx] -= collision_offset
            constraint_marker.set_position(test_position)
            simulator.step()
            
            if len(get_pt_idx_within_object_link(test_position.reshape(1, 3), selected_obj, link, aabb_offset=aabb_offset)) == 0: # apply offset since bbox might cover handle
                body1 = constraint_marker.get_body_id()
                body2 = selected_obj.get_body_id()
                max_distance = 0    
                collision = len(
                    p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=-1, linkIndexB=link, distance=max_distance)
                ) != 0  # getContactPoints
                if not collision:
                    ENOUGH_SPACE_B = True
            
            ENOUGH_SPACE_FOR_HANDLE = ENOUGH_SPACE_A and ENOUGH_SPACE_B
        
        
        else: # prismatic
            ENOUGH_SPACE_FOR_HANDLE = False
            
            # first position
            constraint_marker.set_position([0., 0., 100.])
            restoreState(pb_initial_state)
            test_position = pt.copy()
            test_position += screw_axis * collision_offset
            constraint_marker.set_position(test_position)                    
            simulator.step()
            
            if len(get_pt_idx_within_object_link(test_position.reshape(1, 3), selected_obj, link, aabb_offset=0.0)) == 0: # no offset
                body1 = constraint_marker.get_body_id()
                body2 = selected_obj.get_body_id()
                max_distance = 0    
                collision = len(
                    p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=-1, linkIndexB=link, distance=max_distance)
                ) != 0  # getContactPoints                 
                if not collision:
                    ENOUGH_SPACE_FOR_HANDLE = True
        
        
        # save
        if ENOUGH_SPACE_FOR_HANDLE:
            pos_pt_list.append(pt) 
            pos_pt_idx_list.append(pts_idx_list_on_link[idx_pt_on_link])
        else:
            neg_pt_list.append(pt)    
            neg_pt_idx_list.append(pts_idx_list_on_link[idx_pt_on_link])
        # end of [pts_on_link]
    
    
    return pos_pt_list, neg_pt_list, pos_pt_idx_list, neg_pt_idx_list