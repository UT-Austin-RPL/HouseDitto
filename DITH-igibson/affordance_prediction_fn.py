from copy import deepcopy
import numpy as np
import open3d as o3d
from igibson.external.pybullet_tools.utils import get_all_links, get_joint_type, get_joint_info
from utils.utils_object import get_pt_idx_within_object_link, sample_bbox_in_wf


def check_exists_in_groundtrth(obj_id, link, unique_obj_id_link_id):
    for v in unique_obj_id_link_id:
        [oi, li] = v        
        if obj_id == oi and link == li:
            return True        
    return False


def get_pts_obj_link_id_v2(simulator, unique_obj_id_link_id, pts_opengl, pts_all_pos_opengl, object_list, init_link_dict={}):
    
    obj_link_dict = {}
    obj_link_dict['None'] = {}
    obj_link_dict['None']['None'] = deepcopy(init_link_dict) # not on obj    
    
    pts_obj_link_id = np.full((len(pts_opengl), 2), -1)
    
    for idx_obj, object_name in enumerate(object_list):
        selected_obj = simulator.scene.objects_by_name[object_name]
        obj_id = selected_obj.get_body_id()
        
        for link in get_all_links(obj_id):            
            # Check if base link
            # Check if exists in groundtruth            
            # Check if revolute or prismatic
            if link == -1 \
                or not check_exists_in_groundtrth(obj_id, link, unique_obj_id_link_id) \
                or not get_joint_type(selected_obj.get_body_id(), link) in [0, 1]:
                
                pts_idx_list_on_link = get_pt_idx_within_object_link(pts_opengl, selected_obj, link)
                pts_obj_link_id[pts_idx_list_on_link, 0] = -2 # ignore (not count as false positive)
                pts_obj_link_id[pts_idx_list_on_link, 1] = -2 # ignore (not count as false positive)
                
                continue
            else:
                pts_idx_list_on_link = get_pt_idx_within_object_link(pts_opengl, selected_obj, link)
                pts_obj_link_id[pts_idx_list_on_link, 0] = selected_obj.get_body_id()  
                pts_obj_link_id[pts_idx_list_on_link, 1] = link
            
            # Update obj_link_dict
            if object_name not in obj_link_dict:
                obj_link_dict[object_name] = {}
            obj_link_dict[object_name][str(link)] = deepcopy(init_link_dict)
            obj_link_dict[object_name][str(link)]['object_id'] = selected_obj.get_body_id()
            obj_link_dict[object_name][str(link)]['object_name'] = object_name
            obj_link_dict[object_name][str(link)]['link_id'] = link      
            obj_link_dict[object_name][str(link)]['jointType'] = get_joint_type(selected_obj.get_body_id(), link)
            obj_link_dict[object_name][str(link)]['jointUpperLimit'] = get_joint_info(selected_obj.get_body_id(), link).jointUpperLimit      
            
            
    return pts_obj_link_id, obj_link_dict


def get_pts_obj_link_id_empty_v2(simulator, unique_obj_id_link_id, object_list, init_link_dict={}):
    
    obj_link_dict = {}
    obj_link_dict['None'] = {}
    obj_link_dict['None']['None'] = deepcopy(init_link_dict) # not on obj    
    
    for idx_obj, object_name in enumerate(object_list):        
        selected_obj = simulator.scene.objects_by_name[object_name]
        obj_id = selected_obj.get_body_id()
        
        for link in get_all_links(obj_id):
            
            # Check if base link
            if link == -1:
                continue
                        
            # Check if exists in groundtruth
            if not check_exists_in_groundtrth(obj_id, link, unique_obj_id_link_id):
                continue
            
            # Check if revolute or prismatic
            if not get_joint_type(selected_obj.get_body_id(), link) in [0, 1]:
                continue
            
            # Update obj_link_dict
            if object_name not in obj_link_dict:
                obj_link_dict[object_name] = {}
            obj_link_dict[object_name][str(link)] = deepcopy(init_link_dict)
            obj_link_dict[object_name][str(link)]['object_id'] = selected_obj.get_body_id()
            obj_link_dict[object_name][str(link)]['object_name'] = object_name
            obj_link_dict[object_name][str(link)]['link_id'] = link      
            obj_link_dict[object_name][str(link)]['jointType'] = get_joint_type(selected_obj.get_body_id(), link)
            obj_link_dict[object_name][str(link)]['jointUpperLimit'] = get_joint_info(selected_obj.get_body_id(), link).jointUpperLimit      
            
            
    return obj_link_dict