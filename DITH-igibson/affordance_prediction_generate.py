import os
import json
import numpy as np
import pybullet as p
from pathlib import Path

from igibson import object_states
from igibson.objects.visual_marker import VisualMarker
from igibson.external.pybullet_tools.utils import (
    get_joint_position,
    get_joint_type,
)    
from igibson.utils.utils import restoreState

from misc.args import get_global_args, get_move_direction_list
from utils.utils_scene import create_new_simulator, interact_with_obj_strict
from utils.utils_eval import pts_nms
from affordance_prediction_fn import get_pts_obj_link_id_v2, get_pts_obj_link_id_empty_v2


pointnet_exp_name = "2023-07-27_15-28-25" # the experiment name, e.g., '2023-07-27_15-28-25', found under the path '../DITH-pointnet/logs/experiments/runs/default/'
results_path = "../DITH-pointnet/logs/experiments/runs/default/%s/results" % pointnet_exp_name
groundtruth_path = "../dataset/cubicasa5k_rooms"
output_path = "../pointnet-pred/%s/affordance-first-round/" % pointnet_exp_name
if not os.path.exists(output_path):
    os.makedirs(output_path)


nms_threshold_dist = 0.01   # the distance threshold of NMS
nms_threshold_conf = 0.5    # the confidence threshold of NMS


fname_list = [o for o in os.listdir(results_path)
                if not os.path.isdir(os.path.join(results_path,o))]
fname_list.sort()
fname_list_filtered = fname_list

if os.path.exists(os.path.join(output_path, 'result_threshold_dist_%.4f_conf_%.4f.npz' % (nms_threshold_dist, nms_threshold_conf))):
    save_dict = np.load(os.path.join(output_path, 'result_threshold_dist_%.4f_conf_%.4f.npz' % (nms_threshold_dist, nms_threshold_conf)), allow_pickle=True)
    save_dict = dict(save_dict)
else:
    save_dict = {}


# simulator setting
mode = "headless" # "headless" or "gui_interactive"
use_pb_gui = False

# global setting
global_args = get_global_args()

# interaction
max_force = 50
max_step = 20
strict_move_direction = True
strict_interaction = True


s = None
previous_scene_name = None

for fname in fname_list_filtered:
    key = fname[:-4]
    
    # check if exists
    if key in save_dict:
        continue
    
    # load groundtruth    
    npy_gt = np.load(os.path.join(groundtruth_path, fname), allow_pickle=True)
    unique_obj_id_link_id, counts = np.unique(
        np.stack([npy_gt['pt_object_id'], npy_gt['pt_link_id']], axis=1),
        axis=0,
        return_counts=True
    )
    unique_obj_id_link_id = unique_obj_id_link_id[counts > 100]
    
    # load pointnet prediction
    npy = np.load(os.path.join(results_path, fname), allow_pickle=True)
    pts = npy['pts_sampled']
    pts_opengl = np.stack([pts[:, 2], pts[:, 0], pts[:, 1]], axis=-1) # transform into opengl frame
    # color_list = npy['color_sampled']
    # label_list = npy['label_sampled']    
    affordance = npy['affordance_sampled']
    logit_list = npy['logit_sampled']
    pred_list = npy['pred_sampled']
    
    pts_all = npy['pts']
    pts_all_opengl = np.stack([pts_all[:, 2], pts_all[:, 0], pts_all[:, 1]], axis=-1) # transform into opengl frame
    pts_all_pos_opengl = pts_all_opengl[npy['affordance'].reshape(-1) == 1]
    
    # apply NMS    
    pts_opengl, pred_list, logit_list, affordance = \
        pts_nms(pts_opengl, pred_list, logit_list, affordance, threshold_dist=nms_threshold_dist, threshold_confidence=nms_threshold_conf)
    

    # initialize the simulator on new scene
    scene_name = '%s_%s_%s' % tuple(Path(fname).stem.split('_')[:3])
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
    
    # print object
    for name in s.scene.objects_by_name:
        selected_obj = s.scene.objects_by_name[name]
        print(name, selected_obj.get_body_id())
    del name, selected_obj
    
    # get room information
    room_info = npy['room_info'].item()
    room_ins_id = room_info["room_ins_id"]
    room_ins_name = room_info["room_ins_name"]
    object_list = room_info['room_object_name'][0]
    
    # get object_id/link_id and pts mapping
    init_link_dict = {
        'object_id': None,
        'object_name': None,
        'link_id': None,
        'pts': [],
        'move_direction': [],
        'joint_position': [],
    }
    if pts_opengl is None: # no point left after nms
        obj_link_dict = get_pts_obj_link_id_empty_v2(s, unique_obj_id_link_id, object_list, init_link_dict=init_link_dict)
        obj_link_dict['meta'] = {
            "scene_name": scene_name,
            "room_ins_id": room_ins_id,
            "room_ins_name": room_ins_name,
            "max_force": max_force,
            "max_step": max_step,
            "strict_move_direction": strict_move_direction,
            "strict_interaction": strict_interaction,
        }
        # save
        save_dict[key] = obj_link_dict
        continue
    else:
        pts_obj_link_id, obj_link_dict = get_pts_obj_link_id_v2(s, unique_obj_id_link_id, pts_opengl, pts_all_pos_opengl, object_list, init_link_dict=init_link_dict)
    
        
    # Save pybullet state (kinematics)
    pb_initial_state = p.saveState()
    
    # iterate over all pts
    move_direction_list = get_move_direction_list(strict=strict_move_direction)
    for idx_pt_interaction, (pt, pt_id) in enumerate(zip(pts_opengl, pts_obj_link_id)):
        
        object_id = int(pt_id[0])
        link = int(pt_id[1])
        
        # pts not on obj
        if object_id == -1 or link == -1: # -1: false positive; -2: ignore
            obj_link_dict['None']['None']['pts'].append(pt)
        # pts on obj
        elif object_id <= -2 or link <= -2: # -1: false positive; -2: ignore
            continue
        else:
            obj_link_dict[s.scene.objects_by_id[object_id].name][str(link)]['pts'].append(pt)
                        
            # interact
            selected_obj = s.scene.objects_by_id[object_id]
            max_move_direction = None
            max_joint_position = 0.001
            
            for idx_move_direction, move_direction in enumerate(move_direction_list):

                restoreState(pb_initial_state)
                
                # interact
                interact_with_obj_strict(
                    simulator=s, 
                    obj=selected_obj, 
                    link_id=link, 
                    ball=constraint_marker, 
                    hit_pos=pt,
                    move_direction=move_direction,
                    max_force=max_force,
                    max_step=max_step,
                    strict=strict_interaction,
                )
                
                # determine if opened
                if selected_obj.states[object_states.Open].get_value():
                    joint_position = get_joint_position(selected_obj.get_body_id(), link)
                    joint_type = get_joint_type(selected_obj.get_body_id(), link)
                    assert joint_type in [0, 1]
                    
                    print('Open', joint_type, joint_position, move_direction)
                
                    if np.abs(joint_position) > max_joint_position:
                        max_joint_position = joint_position
                        max_move_direction = move_direction
                else:
                    print('Not Open (obj: %d, link_id: %d)' % (selected_obj.get_body_id(), link))                
                
            # Success
            if max_joint_position != 0.001 and max_move_direction is not None:
                obj_link_dict[s.scene.objects_by_id[object_id].name][str(link)]['move_direction'].append(max_move_direction)
                obj_link_dict[s.scene.objects_by_id[object_id].name][str(link)]['joint_position'].append(max_joint_position)
            # Non-success
            else:
                obj_link_dict[s.scene.objects_by_id[object_id].name][str(link)]['move_direction'].append(np.array([None, None, None]))
                obj_link_dict[s.scene.objects_by_id[object_id].name][str(link)]['joint_position'].append(0.)
    
    # remove ball robot
    constraint_marker.set_position([0., 0., 100.])  
    
    # reset scene
    restoreState(pb_initial_state)
    p.removeState(pb_initial_state)
    
    # convert pts move_dir joint_pos into numpy array    
    for obj_name in obj_link_dict:
        for link in obj_link_dict[obj_name]:
            if len(obj_link_dict[obj_name][str(link)]["pts"]) > 0:
                obj_link_dict[obj_name][str(link)]["pts"] = np.vstack(obj_link_dict[obj_name][str(link)]["pts"])
                if str(link) != 'None': # no 'move_direction' and 'joint_position'
                    obj_link_dict[obj_name][str(link)]["move_direction"] = np.vstack(obj_link_dict[obj_name][str(link)]["move_direction"])
                    obj_link_dict[obj_name][str(link)]["joint_position"] = np.array(obj_link_dict[obj_name][str(link)]["joint_position"])
    
    # add room meta data
    obj_link_dict['meta'] = {
        "scene_name": scene_name,
        "room_ins_id": room_ins_id,
        "room_ins_name": room_ins_name,
    }
    
    # save
    save_dict[key] = obj_link_dict
    
    # save
    np.savez_compressed(os.path.join(output_path, 'result_threshold_dist_%.4f_conf_%.4f.npz' % (nms_threshold_dist, nms_threshold_conf)), **save_dict)