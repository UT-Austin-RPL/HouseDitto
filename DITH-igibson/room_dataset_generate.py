import os
import numpy as np
import pybullet as p
import igibson
from igibson import object_states
from objects.collision_marker import CollisionMarker
from igibson.external.pybullet_tools.utils import (
    get_all_links,
    get_joint_type,
)    
from igibson.utils.utils import restoreState

from utils.utils_misc import make_dirs
from utils.utils_object import get_pt_idx_within_object, get_pt_idx_within_object_link
from utils.utils_render import generate_scene_pointcloud
from utils.utils_scene import create_new_simulator, get_all_room_info
from misc.args import get_global_args
from room_dataset_fn import update_problem_list, filter_cubicasa_scene_list, check_collision_for_gripper


import random
random.seed(42)
rng = np.random.default_rng(42)


# simulator setting
mode = "headless" # "headless" or "gui_interactive"
use_pb_gui = False

# data path
scene_dir = os.path.join(igibson.cubicasa_dataset_path, 'scenes')
save_dir = '../dataset/cubicasa5k_rooms/'
scene_failed_dir = '../dataset/cubicasa5k_rooms_failed/'
make_dirs(save_dir)
make_dirs(scene_failed_dir)

# get scene list
scene_list_filtered = filter_cubicasa_scene_list(scene_dir, save_dir=save_dir, scene_failed_dir=scene_failed_dir)
random.shuffle(scene_list_filtered)

# global setting
global_args = get_global_args()

# get problem list
if os.path.exists('problematic_scene_list'):
    problematic_scene_list = list(np.load('problem_list.npz', allow_pickle=True)['arr_0'])
else:
    problematic_scene_list = []
    


s = None

for scene_idx, scene_name in enumerate(scene_list_filtered):                
    
    global_args['scene_args']['scene_id'] = scene_name
    
    # get collision list
    s = create_new_simulator(simulator=s, mode=mode, use_pb_gui=False, scene_args=global_args['scene_args'])
    body_collision_set, link_collision_set = s.scene.body_collision_set, s.scene.link_collision_set
    # recreate scene to avoid collisions
    s = create_new_simulator(simulator=s, mode=mode, use_pb_gui=use_pb_gui, scene_args=global_args['scene_args'], body_collision_set=body_collision_set, link_collision_set=link_collision_set)
    
    if object_states.Open not in s.scene.objects_by_state:
        problematic_scene_list = update_problem_list(problematic_scene_list, {'name': scene_name}, msg='no valid object')
        continue
    
    # collect room information
    room_info = get_all_room_info(s.scene, global_args['ignore_category'], visualization=True)    
    
    # sample pointcloud
    for room_ins_id, room_ins_info in room_info.items():
        
        # save info
        room_ins_name = room_ins_info['room_ins_name']
        key = '%s_%s_%d' % (scene_name, room_ins_name, room_ins_id)
        save_path = os.path.join(save_dir, key + '.npz')
        
        # check if exists
        if os.path.exists(save_path):
            print('Already exists: %s' % save_path)
            continue
        
        # check if any valid object
        if len(room_ins_info['objects']) <= 0:
            problematic_scene_list = update_problem_list(problematic_scene_list, {'name': key}, msg='no valid object')
            continue
        
        
        new_pts, new_color, new_label, camera_pose_list = generate_scene_pointcloud(s, room_ins_info, sizeof_voxel_pc=global_args['sizeof_voxel_pc'], n_views='8')
            
        # reorder (xzy -> xyz yxz)
        pts_opengl = np.stack([new_pts[:, 2], new_pts[:, 0], new_pts[:, 1]], axis=-1) # transform into opengl frame
        
        # Sample object point cloud
        object_list = room_ins_info['objects']
        saved_dict = {}
        saved_dict['pt_object_id'] = np.full((len(pts_opengl), ), -1)
        saved_dict['pt_link_id'] = np.full((len(pts_opengl), ), -1)
        saved_dict['pt_move_direction'] = np.full((len(pts_opengl), 3), None)
        saved_dict['pt_joint_position'] = np.full((len(pts_opengl), ), 0)
        
        # import ball robot
        constraint_marker = CollisionMarker(radius=global_args['ball_robot_radius'])
        s.import_object(constraint_marker)
        constraint_marker.set_position([0., 0., 100.])
        
        # Save pybullet state (kinematics)
        pb_initial_state = p.saveState()
        
        ##########################################################################
        ################### (1): Sample points on object link ####################
        ##########################################################################

        # initialization
        pos_pt_list = []
        pos_pt_idx_list = []
        neg_pt_list = []
        neg_pt_idx_list = []
        selected_obj = None
        
        for idx_obj, selected_obj in enumerate(object_list):
            for link in get_all_links(selected_obj.get_body_id()):
                
                # Check if base link
                if link == -1:
                    continue
                
                # Check if revolute or prismatic
                if not get_joint_type(selected_obj.get_body_id(), link) in [0, 1]:
                    continue
                

                print("scene_name, room_ins_name, obj_name, idx_obj, link", scene_name, room_ins_name, selected_obj.name, idx_obj, link)

                # sample pts on link
                pts_idx_list_on_link = get_pt_idx_within_object_link(pts_opengl, selected_obj, link, aabb_offset=global_args['aabb_offset'])                
                pts_on_link = pts_opengl[pts_idx_list_on_link]

                # save
                saved_dict['pt_object_id'][pts_idx_list_on_link] = selected_obj.get_body_id()
                saved_dict['pt_link_id'][pts_idx_list_on_link] = link
                
                # check if enough number of pts_on_link
                if len(pts_on_link) <= 100:
                    continue # skip the link
                
                # check collision for gripper
                cur_pos_pt_list, cur_neg_pt_list, cur_pos_pt_idx_list, cur_neg_pt_idx_list = \
                    check_collision_for_gripper(
                        pts_on_link,
                        #
                        s,
                        selected_obj, 
                        link, 
                        constraint_marker, 
                        pb_initial_state, 
                        #
                        pts_idx_list_on_link,
                        #
                        collision_offset=global_args["collision_offset"],
                    )
                
                # save
                pos_pt_list.extend(cur_pos_pt_list)
                neg_pt_list.extend(cur_neg_pt_list)
                pos_pt_idx_list.extend(cur_pos_pt_idx_list)
                neg_pt_idx_list.extend(cur_neg_pt_idx_list)
        
        
        # remove ball robot
        constraint_marker.set_position([0., 0., 100.])  
        
        # reset scene
        restoreState(pb_initial_state)
        p.removeState(pb_initial_state)
        
        # save
        saved_dict['pos_idx'] = pos_pt_idx_list
        saved_dict['neg_link_idx'] = neg_pt_idx_list
        
        del pos_pt_list, pos_pt_idx_list, neg_pt_list, neg_pt_idx_list
        
        ##########################################################################
        ############ (2): Sample points on object but not on any link ############
        ##########################################################################
        
        neg_pt_list = []
        neg_pt_idx_list = []
        
        for idx_obj, selected_obj in enumerate(object_list):
            
            pts_idx_list_on_link = get_pt_idx_within_object_link(pts_opengl, selected_obj, link_id=None, aabb_offset=global_args['aabb_offset'])
            pts_idx_list_on_obj = get_pt_idx_within_object(pts_opengl, selected_obj, aabb_offset=global_args['aabb_offset'])
                        
            pts_idx_list_on_obj_not_on_link = set(pts_idx_list_on_obj) - set(pts_idx_list_on_link)
            pts_idx_list_on_obj_not_on_link = list(pts_idx_list_on_obj_not_on_link)
            pts_on_obj_not_on_link = pts_opengl[pts_idx_list_on_obj_not_on_link]
            
            # Save
            neg_pt_list.extend(pts_on_obj_not_on_link)            
            neg_pt_idx_list.extend(pts_idx_list_on_obj_not_on_link)
            saved_dict['pt_object_id'][pts_idx_list_on_obj_not_on_link] = selected_obj.get_body_id()
            
        # Save
        saved_dict['neg_obj_idx'] = neg_pt_idx_list # `neg_pt_list` is for debugging only        
        del neg_pt_list, neg_pt_idx_list
        
        
        ##########################################################################
        ##################### (4): Postprocess Pointcloud ########################
        ##########################################################################
        
        pos_pt_idx_list = saved_dict['pos_idx']
        neg_pt_idx_list = []
        for k in ['neg_link_idx', 'neg_obj_idx']:
            neg_pt_idx_list.extend(saved_dict[k])
        
        # Check pos and neg exist
        if len(pos_pt_idx_list) <= 10:
            problematic_scene_list = update_problem_list(problematic_scene_list, {'name': key}, msg="no enough positive sample point for %s (link %d): " % (selected_obj.name, link))
            continue
        if len(neg_pt_idx_list) <= 10:
            problematic_scene_list = update_problem_list(problematic_scene_list, {'name': key}, msg="no enough negative sample point for %s (link %d): " % (selected_obj.name, link))
            continue
                    
        ##########################################################################
        ############################ (5): Save Dict ##############################
        ##########################################################################
        
        # meta info
        saved_dict['scene_name'] = scene_name
        saved_dict['room_name'] = room_ins_name
        # saved_dict['object_name'] = obj.name
        saved_dict['object_id_to_name'] = {k:v.name for k, v in s.scene.objects_by_id.items() if hasattr(v, "name")}
        saved_dict['object_name_to_id'] = {v.name: k for k, v in s.scene.objects_by_id.items() if hasattr(v, "name")}
        saved_dict['room_info'] = {k: v for k, v in room_ins_info.items() if not k in ['objects', 'objects_bbox']}
        saved_dict['room_info']['room_object_name'] = [obj.name for obj in room_ins_info['objects']],
        saved_dict['room_info']['room_object_category'] = [obj.category for obj in room_ins_info['objects']]
        saved_dict['global_args'] = global_args
        
        # pointcloud
        saved_dict['pts'] = new_pts
        saved_dict['color'] = new_color.reshape(-1, 3)
        saved_dict['label'] = new_label.reshape(-1, 1)
        saved_dict['affordance'] = np.full((len(new_pts), 1), -1)
        saved_dict['affordance'][pos_pt_idx_list] = 1
        saved_dict['affordance'][neg_pt_idx_list] = 0
        
        # save dict
        np.savez_compressed(save_path, **saved_dict)