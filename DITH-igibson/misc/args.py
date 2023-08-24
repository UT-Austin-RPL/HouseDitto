import numpy as np


def get_global_args():

    global_args = {}

    # scene setting
    not_load_object_categories = ['ceilings', 'walls', 'door', 'window', 'top_cabinet']
    global_args['scene_args'] = {
        "scene_id": None,
        "scene_source": "CUBICASA",
        "not_load_object_categories": not_load_object_categories,
    }

    # point cloud sampling
    global_args['ignore_category'] = ['door', 'window', 'top_cabinet']
    global_args['sizeof_voxel_pc'] = [0.015, 0.015, 0.015] #[0.025, 0.025, 0.025]
    
    
    # affordance sampling
    global_args['aabb_offset'] = 0.0 # offset for sampling bbox
    global_args['sizeof_voxel_affordance'] = [0.1, 0.1, 0.1]
    global_args['revolute_openness_pos_threshold'] = np.pi / 6.
    # global_args['revolute_openness_neg_threshold'] = 0. #np.pi / 48.
    global_args['prismatic_openness_pos_threshold'] = 0.1
    # global_args['prismatic_openness_neg_threshold'] = 0. #0.05
    global_args['ball_robot_radius'] = 1e-4
    global_args['collision_offset'] = 0.03
    
    return global_args


def get_move_direction_list(strict=False):
    if strict:
        move_direction_list = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],  
            [0, -1, 0],
            [0, 0, 1],
            # [0, 0, -1],
        ])        
    else:
        move_direction_list = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],  
            [0, -1, 0],
            [0, 0, 1],
            # [0, 0, -1],
            #
            [1, 1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            #
            [0, 1, 1],
            [0, -1, 1],
            # [0, 1, -1],
            # [0, -1, -1],
            #
            [1, 0, 1],
            [-1, 0, 1],
            # [1, 0, -1],
            # [-1, 0, -1],
        ])
    return move_direction_list