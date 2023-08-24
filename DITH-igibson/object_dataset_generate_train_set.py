import os
import glob
import numpy as np
rng = np.random.default_rng(42)

import pybullet as p

from igibson import object_states
from igibson.objects.visual_marker import VisualMarker
from igibson.external.pybullet_tools.utils import (
    get_all_links,
    get_joint_position,
    get_joint_type,
    get_joint_info,
)    
from igibson.utils.utils import restoreState

from misc.args import get_global_args
from object_dataset_fn import get_best_move_direction, get_lists, save_lists
from utils.utils_misc import make_dirs, update_problem_list
from utils.utils_object import get_link_bbox_center, get_joint_screw, get_pt_idx_within_object_link
from utils.utils_render import generate_partial_pointcloud
from utils.utils_scene import create_new_simulator, get_all_room_info, interact_with_obj_strict



train_scene_list_path = '../dataset/cubicasa5k_rooms_processed/split/train_scene.txt'
aff_npy_path = '../dataset/cubicasa5k_rooms'
output_path = '../dataset/cubicasa5k_objects/train'



meta_path = os.path.join(output_path, 'meta')
make_dirs(meta_path)


# split subset from affordance's training set
with open(train_scene_list_path, 'r') as f:
    scene_list = f.readlines()
    scene_list = [l.rstrip() for l in scene_list]

with open(os.path.join(meta_path, 'train_scene.txt'), 'w') as f:
    f.writelines('\n'.join(scene_list))


# simulator setting
mode = "headless" # "headless" or "gui_interactive"
use_pb_gui = False

# get problem list
filtered_scene_list, problematic_scene_list, generate_ditto_data_scene_list = get_lists(meta_path, scene_list)

# global setting
global_args = get_global_args()

# interaction
max_force = 99
max_step = 99


s = None
previous_scene_name = None


for scene_room_name in filtered_scene_list:
    
    # get scene and room name
    scene_name = '%s_%s_%s' % tuple(scene_room_name.split('_')[:3])
    room_name = '_'.join(tuple(scene_room_name[:-4].split('_')[3:-1])) # remove room_id
        
    # create simulator
    if not previous_scene_name == scene_name:
        
        global_args['scene_args']['scene_id'] = scene_name
        
        # get collision list
        s = create_new_simulator(simulator=s, mode=mode, use_pb_gui=False, scene_args=global_args['scene_args'])
        body_collision_set, link_collision_set = s.scene.body_collision_set, s.scene.link_collision_set
        # recreate scene to avoid collisions
        s = create_new_simulator(simulator=s, mode=mode, use_pb_gui=use_pb_gui, scene_args=global_args['scene_args'], body_collision_set=body_collision_set, link_collision_set=link_collision_set)

        # import ball robot
        constraint_marker = VisualMarker(radius=0.01, rgba_color=[0, 0, 1, 1])
        s.import_object(constraint_marker)
        constraint_marker.set_position([0., 0., 100.])
    
        # collect room information
        room_info = get_all_room_info(s.scene, global_args['ignore_category'], visualization=True)   
        
        # save pybullet state (kinematics)
        pb_initial_state = p.saveState()

    print(room_name)
    print([v['room_ins_name'] for k,v in room_info.items()])

    assert room_name in [v['room_ins_name'] for k,v in room_info.items()]
    for room_ins_id, room_ins_info in room_info.items():
        
        # check if the name match
        if room_name != room_ins_info['room_ins_name']:
            continue
        
        # check if any valid object
        if len(room_ins_info['objects']) <= 0:
            # problematic_scene_list = update_problem_list(problematic_scene_list, {'name': key}, msg='no valid object')
            continue
        
        for obj in room_ins_info['objects']:
            for link in get_all_links(obj.get_body_id()):
                
                # check if base
                if link == -1:
                    continue
                
                # check if revolute or prismatic
                if not get_joint_type(obj.get_body_id(), link) in [0,1]:
                    continue
                
                # check if jointUpperLimit > 0.
                if get_joint_info(obj.get_body_id(), link).jointUpperLimit <= 0.:
                    continue
                
                # get saved_name
                saved_name = '%s_%s_%d %s %d' % (scene_name, room_ins_info['room_ins_name'], room_ins_id, obj.name, link)
                print(saved_name)
                
                # check if exists
                if os.path.exists(os.path.join(output_path, '%s.npz' % saved_name)):
                    generate_ditto_data_scene_list.append(saved_name)
                    continue
                
                # get joint info
                joint_index = int(link)
                join_type = get_joint_type(obj.get_body_id(), link)
                jointUpperLimit = get_joint_info(obj.get_body_id(), link).jointUpperLimit
                jointLowerLimit = get_joint_info(obj.get_body_id(), link).jointLowerLimit
                
                # get random start and end state
                # range_lim = jointUpperLimit - jointLowerLimit
                # range_scale = 0.3                
                # move_range = np.random.uniform(range_lim * range_scale, range_lim)
                # start_state = np.random.uniform(jointLowerLimit, jointUpperLimit - move_range)
                # end_state = start_state + move_range
                
                # get link center
                link_bbox, link_center = get_link_bbox_center(obj, link)
                
                # get hit pos
                aff_npy = np.load(os.path.join(aff_npy_path, scene_room_name), allow_pickle=True)
                pos_pts = aff_npy['pts'][aff_npy['affordance'].reshape(-1) == 1]
                pos_pts_opengl = np.stack([pos_pts[:, 2], pos_pts[:, 0], pos_pts[:, 1]], axis=-1) # transform into opengl frame
                
                selected_idx = get_pt_idx_within_object_link(pos_pts_opengl, obj, link)
                pos_pts_on_link = pos_pts_opengl[selected_idx]
                # move_direction = aff_npy['pt_move_direction'][selected_idx]
                if len(pos_pts_on_link) <= 0:
                    generate_ditto_data_scene_list.append(saved_name)
                    problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no pos point on link')
                    continue
                
                selected_idx = rng.integers(len(pos_pts_on_link))
                hit_pos = pos_pts_on_link[selected_idx]
                # move_direction = move_direction[selected_idx]
                
                # reset
                p.changeDynamics(obj.get_body_id(), -1, mass=0, activationState=p.ACTIVATION_STATE_WAKE_UP) # make sure the renderer is updated
                restoreState(pb_initial_state)
                for _ in range(30):
                    s.step()
                
                # set to start state
                # set_joint_position(obj.get_body_id(), link, start_state)
                # for _ in range(30):
                #     s.step()
                
                # create before obj partial pc
                pts, color, label, camera_pose_list = generate_partial_pointcloud(link_center, s, room_ins_info)                

                # check if no valid point
                if pts is None:
                    generate_ditto_data_scene_list.append(saved_name)
                    problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no valid point (before)')
                    continue
                
                # save pc_start
                pc_start = pts
                color_start = color
                label_start = label
                state_start = p.getJointState(obj.get_body_id(), link)[0]
                
                # get part segmentation label
                pts_opengl = np.stack([pts[:, 2], pts[:, 0], pts[:, 1]], axis=-1) # transform into opengl frame
                pc_seg_label_start = np.full((len(pts), ), 0)
                pts_idx_list_on_link = get_pt_idx_within_object_link(pts_opengl, obj, link)
                pc_seg_label_start[pts_idx_list_on_link] = 1
                if len(pts_idx_list_on_link) <= 10:
                    generate_ditto_data_scene_list.append(saved_name)
                    problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no enough on link point (before)')
                    continue
                              
                # reset
                p.changeDynamics(obj.get_body_id(), -1, mass=0, activationState=p.ACTIVATION_STATE_WAKE_UP) # make sure the renderer is updated
                restoreState(pb_initial_state)
                for _ in range(30):
                    s.step()
                
                # set to end state
                # set_joint_position(obj.get_body_id(), link, end_state)
                # for _ in range(30):
                #     s.step()
                
                # get best move_direction
                joint_type = get_joint_type(obj.get_body_id(), link)
                openness_pos_threshold = global_args['revolute_openness_pos_threshold'] if joint_type == 0 else global_args['prismatic_openness_pos_threshold']
                best_move_direction, best_joint_position = get_best_move_direction(s, obj, link, hit_pos, constraint_marker, pb_initial_state, max_force=max_force, max_step=max_step)
                
                # check if valid move_direction
                if best_move_direction is None or best_joint_position <= openness_pos_threshold:
                    generate_ditto_data_scene_list.append(saved_name)
                    problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no valid move_direction')
                    continue



                # reset
                p.changeDynamics(obj.get_body_id(), -1, mass=0, activationState=p.ACTIVATION_STATE_WAKE_UP) # make sure the renderer is updated
                restoreState(pb_initial_state)
                for _ in range(30):
                    s.step()
                
                # interact
                interact_with_obj_strict(
                    simulator=s, 
                    obj=obj, 
                    link_id=link, 
                    ball=constraint_marker, 
                    hit_pos=hit_pos,
                    move_direction=best_move_direction,
                    max_force=max_force,
                    max_step=max_step,
                )
                    
                # check if opened
                if not obj.states[object_states.Open].get_value():
                    generate_ditto_data_scene_list.append(saved_name)
                    problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no valid move_direction')
                    continue
                joint_type = get_joint_type(obj.get_body_id(), link)
                assert joint_type in [0, 1]
                joint_position = get_joint_position(obj.get_body_id(), link)
                assert joint_position > openness_pos_threshold
                
                # create after obj partial pc
                pts, color, label, camera_pose_list = generate_partial_pointcloud(link_center, s, room_ins_info)
                
                # check if no valid point
                if pts is None:
                    generate_ditto_data_scene_list.append(saved_name)
                    problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no valid point (after)')
                    continue
                
                # save pc_end
                pc_end = pts
                color_end = color
                label_end = label
                state_end = p.getJointState(obj.get_body_id(), link)[0]
                
                # get part segmentation label
                pts_opengl = np.stack([pts[:, 2], pts[:, 0], pts[:, 1]], axis=-1) # transform into opengl frame
                pc_seg_label_end = np.full((len(pts), ), 0)
                pts_idx_list_on_link = get_pt_idx_within_object_link(pts_opengl, obj, link)
                pc_seg_label_end[pts_idx_list_on_link] = 1
                if len(pts_idx_list_on_link) <= 10:
                    generate_ditto_data_scene_list.append(saved_name)
                    problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no enough on link point (after)')
                    continue
                
                # get joint screw            
                screw_axis, screw_moment = get_joint_screw(obj, joint_index)
                screw_axis = np.array([screw_axis[1], screw_axis[2], screw_axis[0]]) # transform from opengl frame into pts frame
                screw_moment = np.array([screw_moment[1], screw_moment[2], screw_moment[0]]) # transform from opengl frame into pts frame
                screw_point = np.cross(screw_axis, screw_moment)
                
                # get normalization parameters
                xyz_min = room_ins_info['xyz_min']
                xyz_max = room_ins_info['xyz_max']
                xyz_max[2] = 2.4 # height
                xyz_min_max = np.vstack([xyz_min, xyz_max])
                xyz_min_max_pts_frame = np.stack([xyz_min_max[:, 1], xyz_min_max[:, 2], xyz_min_max[:, 0]], axis=-1)
                center = np.expand_dims(np.mean(xyz_min_max_pts_frame, axis = 0), 0)
                dist = np.max(np.sqrt(np.sum((xyz_min_max_pts_frame - center) ** 2, axis = 1)),0)
                
                saved_dict = {
                    "scene_name": scene_name,
                    "room_ins_id": room_ins_id,
                    "room_ins_name": room_ins_info['room_ins_name'],
                    "obj_name": obj.name,
                    "object_id": obj.get_body_id(),
                    "link_id": int(link),
                    #
                    "hit_pos": hit_pos,
                    "move_direction": best_move_direction,
                    #
                    "pc_start": pc_start,  # N, 3
                    "pc_end": pc_end,
                    # "pc_start_end": pc_end,
                    # "pc_end_start": pc_start,
                    #
                    "pc_seg_start": pc_seg_label_start,
                    "pc_seg_end": pc_seg_label_end,
                    #
                    "color_start": color_start,
                    "label_start": label_start,
                    "color_end": color_end,
                    "label_start": label_end,
                    #
                    "state_start": state_start,
                    "state_end": state_end,
                    #
                    "screw_axis": screw_axis,
                    "screw_moment": screw_moment,
                    #
                    "joint_type": joint_type, # 0 for revolute joint; 1 for prismatic joint
                    "joint_index": joint_index,
                    #
                    "scale": dist,
                    "center": center,
                }
                
                save_path = os.path.join(output_path, '%s.npz' % saved_name)
                np.savez_compressed(save_path, **saved_dict)
                
                generate_ditto_data_scene_list.append(saved_name)
                
                # save lists
                save_lists(meta_path, problematic_scene_list, generate_ditto_data_scene_list)