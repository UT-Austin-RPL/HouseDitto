import os
import numpy as np
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
from utils.utils_misc import make_dirs, update_problem_list
from utils.utils_object import get_joint_screw, get_pt_idx_within_object_link
from utils.utils_render import generate_partial_pointcloud
from utils.utils_scene import create_new_simulator, get_all_room_info, interact_with_obj_strict
from object_dataset_fn import get_lists

# interaction
max_force = 50
max_step = 20
strict_move_direction = True
strict_interaction = True

pointnet_exp_name = "2023-07-27_15-28-25" # the experiment name, e.g., '2023-07-27_15-28-25', found under the path '../pointnet-pred'
npy_path = '../pointnet-pred/%s/affordance-first-round/result_threshold_dist_0.0100_conf_0.5000.npz' % pointnet_exp_name
output_path = '../dataset/cubicasa5k_objects/test'



meta_path = os.path.join(output_path, 'meta')
make_dirs(meta_path)


# simulator setting
mode = "headless" # "headless" or "gui_interactive"
use_pb_gui = False

# load npy
npy = np.load(os.path.join(npy_path), allow_pickle=True)
key_list = list(npy)
key_list.sort()

# get problem list
filtered_key_list, problematic_scene_list, generate_ditto_data_scene_list = get_lists(meta_path, key_list)

# global setting
global_args = get_global_args()




s = None
previous_scene_name = None


for key in filtered_key_list:
    
    # save generated/skipped scene list
    np.savez_compressed(os.path.join(output_path, 'meta', 'generate_ditto_data_scene_list.npz'), generate_ditto_data_scene_list)    
    # save problem list
    np.savez_compressed(os.path.join(output_path, 'meta', 'problem_list_ditto.npz'), problematic_scene_list)
        
    scene_name = '%s_%s_%s' % tuple(key.split('_')[:3])
    
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
    
    
    for obj_name in npy[key].item():
        
        # ignore meta data
        if obj_name in ['meta', 'None']:
            continue
        
        # get room_ins_info
        room_ins_id = npy[key].item()["meta"]["room_ins_id"]
        room_ins_info = room_info[room_ins_id]
        
                
        for link_id in npy[key].item()[obj_name]:
            
            # filename for saving
            selected_obj = s.scene.objects_by_name[obj_name]
            saved_name = '%s_%s_%d %s %d' % (scene_name, room_ins_info['room_ins_name'], room_ins_id, selected_obj.name, int(link_id))
            
            
            # check if exists
            if os.path.exists(os.path.join(output_path, '%s.npz' % saved_name)):
                generate_ditto_data_scene_list.append(saved_name)
                continue
                
            
            # check if any interaction point
            if len(npy[key].item()[obj_name][link_id]['pts']) <= 0:
                generate_ditto_data_scene_list.append(saved_name)
                problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no interaction point')
                continue
            
            jointUpperLimit = npy[key].item()[obj_name][link_id]['jointUpperLimit']
            joint_position = npy[key].item()[obj_name][link_id]['joint_position']
            
            # get pt and move_direction
            idx_maximum = np.argmax(joint_position)
            pt_maximum = npy[key].item()[obj_name][link_id]['pts'][idx_maximum]
            joint_position_maximum = npy[key].item()[obj_name][link_id]['joint_position'][idx_maximum]
            move_direction_maximum = npy[key].item()[obj_name][link_id]['move_direction'][idx_maximum]
            
            # check if un-openable
            if joint_position_maximum <= 0.:
                generate_ditto_data_scene_list.append(saved_name)
                problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no successful interaction point')
                continue
            
            # reset
            p.changeDynamics(selected_obj.get_body_id(), -1, mass=0, activationState=p.ACTIVATION_STATE_WAKE_UP) # make sure the renderer is updated
            restoreState(pb_initial_state)
            for _ in range(30):
                s.step()
            
            # create before obj partial pc
            pts, color, label, camera_pose_list = generate_partial_pointcloud(pt_maximum, s, room_ins_info)
            
            # check if no valid point
            if pts is None:
                generate_ditto_data_scene_list.append(saved_name)
                problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no valid point (before)')
                continue
            
            # save pc_start
            pc_start = pts
            color_start = color
            label_start = label
            state_start = p.getJointState(selected_obj.get_body_id(), int(link_id))[0]
            
            # get part segmentation label
            pts_opengl = np.stack([pts[:, 2], pts[:, 0], pts[:, 1]], axis=-1) # transform into opengl frame
            pc_seg_label_start = np.full((len(pts), ), 0)
            pts_idx_list_on_link = get_pt_idx_within_object_link(pts_opengl, selected_obj, int(link_id))
            pc_seg_label_start[pts_idx_list_on_link] = 1
            if len(pts_idx_list_on_link) <= 10:
                generate_ditto_data_scene_list.append(saved_name)
                problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no enough on link point (before)')
                continue
            
            # interact
            interact_with_obj_strict(
                simulator=s, 
                obj=selected_obj, 
                link_id=int(link_id), 
                ball=constraint_marker, 
                hit_pos=pt_maximum,
                move_direction=move_direction_maximum,
                max_force=max_force,
                max_step=max_step,
                strict=strict_interaction,
            )
            
            # check if opened
            assert selected_obj.states[object_states.Open].get_value()
            joint_position = get_joint_position(selected_obj.get_body_id(), int(link_id))
            joint_type = get_joint_type(selected_obj.get_body_id(), int(link_id))
            assert joint_type in [0, 1]
                
                
            # create after obj partial pc
            pts, color, label, camera_pose_list = generate_partial_pointcloud(pt_maximum, s, room_ins_info)
            
            # check if no valid point
            if pts is None:
                generate_ditto_data_scene_list.append(saved_name)
                problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no valid point (after)')
                continue
            
            # save pc_end
            pc_end = pts
            color_end = color
            label_end = label
            state_end = p.getJointState(selected_obj.get_body_id(), int(link_id))[0]
            
            # get part segmentation label
            pts_opengl = np.stack([pts[:, 2], pts[:, 0], pts[:, 1]], axis=-1) # transform into opengl frame
            pc_seg_label_end = np.full((len(pts), ), 0)
            pts_idx_list_on_link = get_pt_idx_within_object_link(pts_opengl, selected_obj, int(link_id))
            pc_seg_label_end[pts_idx_list_on_link] = 1
            if len(pts_idx_list_on_link) <= 0:
                generate_ditto_data_scene_list.append(saved_name)
                problematic_scene_list = update_problem_list(problematic_scene_list, {'name': saved_name}, msg='no enough on link point (after)')
                continue
            
            
            # get joint info
            joint_index = int(link_id)
            joint_type = p.getJointInfo(
                selected_obj.get_body_id(), 
                int(link_id)
            )[2]
            assert joint_type in [0, 1]
            
            # get joint screw            
            screw_axis, screw_moment = get_joint_screw(selected_obj, joint_index)
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
                "obj_name": selected_obj.name,
                "object_id": selected_obj.get_body_id(),
                "link_id": int(link_id),
                #
                "hit_pos": pt_maximum,
                "move_direction": move_direction_maximum,
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