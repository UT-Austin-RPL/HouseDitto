import numpy as np
import pybullet as p
import open3d as o3d

from igibson.external.pybullet_tools.utils import get_all_links
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson import object_states

from utils.utils_object import get_joint_screw

from sklearn.metrics.pairwise import euclidean_distances


def init_simulator(simulator, scene_args, body_collision_set=[], link_collision_set=[]):  
    scene = InteractiveIndoorScene(**scene_args)    
    
    # remove collisions
    if object_states.Open in scene.objects_by_state:
        for obj in scene.objects_by_state[object_states.Open]:
            if obj.name in body_collision_set or obj.name in link_collision_set:
                print(obj.name)
                scene.remove_object(obj)    
    
    simulator.import_scene(scene)


def create_new_simulator(mode, use_pb_gui, scene_args, simulator=None, body_collision_set=[], link_collision_set=[]):
        if simulator is not None:
            simulator.scene.reset_scene_objects()
            if object_states.Open in simulator.scene.objects_by_state:
                for selected_obj in simulator.scene.objects_by_state[object_states.Open]:
                    selected_obj.states[object_states.Open].clear_cached_value()
                simulator.step()
            simulator.disconnect()
            
        new_simulator = Simulator(
            mode=mode, 
            use_pb_gui=use_pb_gui, 
            image_width=640,
            image_height=480,
        )
        init_simulator(new_simulator, scene_args, body_collision_set, link_collision_set)
        return new_simulator


def get_all_room_info(scene, ignore_category=[], remove_collisions=True, visualization=True):
    room_info = {}
    for room_ins_name, room_ins_id in scene.room_ins_name_to_ins_id.items():
        
        valid_idx = np.array(np.where(scene.room_ins_map == room_ins_id))
        
        x_min, x_max = np.min(valid_idx[0, :]), np.max(valid_idx[0, :])
        y_min, y_max = np.min(valid_idx[1, :]), np.max(valid_idx[1, :])
        
        x_min, y_min = scene.seg_map_to_world(np.array([x_min, y_min]))
        x_max, y_max = scene.seg_map_to_world(np.array([x_max, y_max]))
        
        xy_min = np.array([x_min, y_min])
        xy_max = np.array([x_max, y_max])
        xyz_min = np.array([x_min, y_min, scene.floor_heights[0]])
        xyz_max = np.array([x_max, y_max, scene.floor_heights[0]])
        
        xy_mid = (xy_min + xy_max) / 2.0
        xyz_mid = (xyz_min + xyz_max) / 2.0
        
        if visualization:
            p.addUserDebugLine(xyz_mid, xyz_mid+0.05, lineWidth=4, lineColorRGB=[1., 1., 1.])  
    
        room_info[room_ins_id] = {
            "room_ins_id": room_ins_id,
            "room_ins_name": room_ins_name,
            "xyz_min": xyz_min,
            "xyz_max": xyz_max,
            "xyz_mid": xyz_mid,
            "xy_min": xy_min,
            "xy_max": xy_max,
            "xy_mid": xy_mid,
            "objects": [], # add later
            "objects_bbox": [], # add later
        }
    
    # collect room object information
    for obj in scene.objects_by_state[object_states.Open]:
        # ignore
        if obj.category in ignore_category:
            continue
        
        # collision
        if remove_collisions:
            if obj.name in scene.body_collision_set or obj.name in scene.link_collision_set:
                continue
        
        # generate object bbox
        boundaries = p.getAABB(obj.get_body_id())
        bbox = o3d.geometry.AxisAlignedBoundingBox()
        bbox = bbox.create_from_points(o3d.utility.Vector3dVector(boundaries))
        
        # assign to room        
        pos = obj.get_position()
        pos_xy = np.array([pos[0], pos[1]])          
        for room_ins_id, room_ins_info in room_info.items():
            if np.all(pos_xy <= room_ins_info["xy_max"]) and np.all(pos_xy >= room_ins_info["xy_min"]):
                room_info[room_ins_id]["objects"].append(obj)
                room_info[room_ins_id]["objects_bbox"].append(bbox)
                break
    
    return room_info


def interact_with_obj(simulator, obj, link_id, ball, hit_pos, move_direction, step_size=0.025):
    object_id = obj.get_body_id()
    posObj = obj.get_position()
    
    # disable occlusion
    for cur_link in get_all_links(obj.get_body_id()):
        p.setCollisionFilterPair(
            ball.get_body_id(), obj.get_body_id(), -1, cur_link, False
        )
    simulator.step()
    
    ball.set_position(hit_pos)
    simulator.step()    
    
    p.changeDynamics(object_id, -1, mass=0, activationState=p.ACTIVATION_STATE_WAKE_UP)
    if link_id == -1:
        link_pos, link_orn = p.getBasePositionAndOrientation(object_id)
    else:
        link_state = p.getLinkState(object_id, link_id)
        link_pos, link_orn = link_state[:2]

    child_frame_trans_pos, child_frame_trans_orn = p.invertTransform(link_pos, link_orn)
    child_frame_pos, child_frame_orn = p.multiplyTransforms(
        child_frame_trans_pos, child_frame_trans_orn, hit_pos, [0, 0, 0, 1]
    )
    
    radius=0.1
    ball.set_position(hit_pos + move_direction * radius)
    
    cid = p.createConstraint(
        parentBodyUniqueId=ball.get_body_id(),
        parentLinkIndex=-1,
        childBodyUniqueId=object_id,
        childLinkIndex=link_id,
        jointType=p.JOINT_POINT2POINT,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=child_frame_pos,
        childFrameOrientation=child_frame_orn,
    )
    p.changeConstraint(cid, maxForce=350)
    simulator.step()
    
    
    
    
    
    # simulator.step()
    for i in range(50):
        new_position = hit_pos + move_direction * step_size * i
        ball.set_position(new_position)
        simulator.step()
        
    # enable occlusion
    for cur_link in get_all_links(obj.get_body_id()):
        p.setCollisionFilterPair(
            ball.get_body_id(), obj.get_body_id(), -1, cur_link, True
        )
    simulator.step()
    
    # reset
    p.removeConstraint(cid)
    ball.set_position([0., 0., 100.])
    simulator.step()


def interact_with_obj_strict(simulator, obj, link_id, ball, hit_pos, move_direction, step_size=0.025, strict=True, max_force=50, max_step=60):
    '''Compare to `interact_with_obj`, this function takes the distance between `hit_pos` and `axis_pos` into consideration'''
    
    object_id = obj.get_body_id()
    posObj = obj.get_position()
    
    # disable occlusion
    for cur_link in get_all_links(obj.get_body_id()):
        p.setCollisionFilterPair(
            ball.get_body_id(), obj.get_body_id(), -1, cur_link, False
        )
    simulator.step()
    
    ball.set_position(hit_pos)
    simulator.step()
    
    p.changeDynamics(object_id, -1, mass=0, activationState=p.ACTIVATION_STATE_WAKE_UP)
    if link_id == -1:
        link_pos, link_orn = p.getBasePositionAndOrientation(object_id)
    else:
        link_state = p.getLinkState(object_id, link_id)
        link_pos, link_orn = link_state[:2]

    child_frame_trans_pos, child_frame_trans_orn = p.invertTransform(link_pos, link_orn)
    child_frame_pos, child_frame_orn = p.multiplyTransforms(
        child_frame_trans_pos, child_frame_trans_orn, hit_pos, [0, 0, 0, 1]
    )
    
    # add an offset to avoid collision (depends on ball size)
    radius=0.1
    ball.set_position(hit_pos + move_direction * radius)
    
    cid = p.createConstraint(
        parentBodyUniqueId=ball.get_body_id(),
        parentLinkIndex=-1,
        childBodyUniqueId=object_id,
        childLinkIndex=link_id,
        jointType=p.JOINT_POINT2POINT,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=child_frame_pos,
        childFrameOrientation=child_frame_orn,
    )
    # compute force
    maxforce = max_force
    
    p.changeConstraint(cid, maxForce=maxforce)
    simulator.step()
    
    # compute steps
    if strict:
        screw_axis, screw_moment = get_joint_screw(obj, link_id)
        point_on_axis = np.cross(screw_axis, screw_moment)
        distance = euclidean_distances(hit_pos[:2].reshape(1, -1), point_on_axis[:2].reshape(1, -1)) # ignore z-axis
        n_step = int(distance * max_step)
    else:
        n_step = max_step
    
    for i in range(n_step):
        new_position = hit_pos + move_direction * step_size * i
        ball.set_position(new_position)
        simulator.step()
    
    # enable occlusion
    for cur_link in get_all_links(obj.get_body_id()):
        p.setCollisionFilterPair(
            ball.get_body_id(), obj.get_body_id(), -1, cur_link, True
        )
    simulator.step()
    
    # reset
    p.removeConstraint(cid)
    ball.set_position([0., 0., 100.])
    simulator.step()


def reset_scene_object(simulator):
    
    # close all link
    for obj in simulator.scene.objects_by_state[object_states.Open]:
        obj.states[object_states.Open].clear_cached_value()
        obj.states[object_states.Open].set_value(False)

    for _ in range(100):
        simulator.step()
        
    # reset scene
    simulator.scene.reset_scene_objects()


def close_irreverent_scene_object_link(simulator, selected_obj, selected_link=None):
    
    # close all obj except selected_obj
    for obj in simulator.scene.objects_by_state[object_states.Open]:
        if obj.get_body_id() != selected_obj.get_body_id():
            
            # set openness as False
            obj.states[object_states.Open].clear_cached_value()
            obj.states[object_states.Open].set_value(False)

    for _ in range(100):
        simulator.step()