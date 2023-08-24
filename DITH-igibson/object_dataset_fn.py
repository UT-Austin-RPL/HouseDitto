import os
import numpy as np
import pybullet as p

from igibson.external.pybullet_tools.utils import (
    get_joint_position,
    get_joint_type,
    set_joint_position,
)    
from igibson.utils.utils import restoreState

from misc.args import get_move_direction_list
from utils.utils_scene import interact_with_obj_strict


def get_lists(meta_path, npy_key_list):
        
    filtered_key_list = npy_key_list
    problematic_scene_list = []
    generate_ditto_data_scene_list = []
    
    if os.path.exists(os.path.join(meta_path, 'problem_list_ditto.npz')):
        # get problematic_scene_list
        problematic_scene_list = list(np.load(os.path.join(meta_path, 'problem_list_ditto.npz'), allow_pickle=True)['arr_0'])
    
    if os.path.exists(os.path.join(meta_path, 'generate_ditto_data_scene_list.npz')):
        # get generate_ditto_data_scene_list
        generate_ditto_data_scene_list = np.load(os.path.join(meta_path, 'generate_ditto_data_scene_list.npz'), allow_pickle=True)['arr_0']
        generate_ditto_data_scene_list = [str(x) for x in generate_ditto_data_scene_list]
        
        if len(generate_ditto_data_scene_list) > 0:
            # get filtered_key_list
            idx_start = 0
            latest_scene = generate_ditto_data_scene_list[-1]
            for idx_key, key in enumerate(npy_key_list):                    
                if latest_scene.startswith(key[:-4]): # remove .npz
                    idx_start = idx_key
                    break
            idx_start = idx_start - 1 if idx_start > 0 else 0
            filtered_key_list = npy_key_list[idx_start:]
    
    return filtered_key_list, problematic_scene_list, generate_ditto_data_scene_list


def save_lists(meta_path, problematic_scene_list, generate_ditto_data_scene_list):
    # save generated/skipped scene list
    np.savez_compressed(os.path.join(meta_path, 'generate_ditto_data_scene_list.npz'), generate_ditto_data_scene_list)    
    # save problem list
    np.savez_compressed(os.path.join(meta_path, 'problem_list_ditto.npz'), problematic_scene_list)


def get_best_move_direction(simulator, obj, link, hit_pos, constraint_marker, pb_initial_state, initial_state=None, strict_move_direction=False, strict_interaction=True, max_force=50, max_step=60):
    best_move_direction = None
    best_joint_position = 0.001
    move_direction_list = get_move_direction_list(strict=strict_move_direction)
    for idx_move_direction, move_direction in enumerate(move_direction_list):
        
        p.changeDynamics(obj.get_body_id(), -1, mass=0, activationState=p.ACTIVATION_STATE_WAKE_UP) # make sure the renderer is updated
        restoreState(pb_initial_state)
        for _ in range(30):
            simulator.step()
        
        if initial_state is not None:
            set_joint_position(obj.get_body_id(), link, initial_state)
            for _ in range(30):
                simulator.step()
        
        # interact
        interact_with_obj_strict(
            simulator=simulator, 
            obj=obj, 
            link_id=link, 
            ball=constraint_marker, 
            hit_pos=hit_pos,
            move_direction=move_direction,
            max_force=max_force,
            max_step=max_step,
            strict=strict_interaction,
        )
        
        joint_position = get_joint_position(obj.get_body_id(), link)
        joint_type = get_joint_type(obj.get_body_id(), link)
        assert joint_type in [0, 1]
        
        if np.abs(joint_position) > np.abs(best_joint_position):
            best_joint_position = joint_position
            best_move_direction = move_direction
    
    return best_move_direction, best_joint_position