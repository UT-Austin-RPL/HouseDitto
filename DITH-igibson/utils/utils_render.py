import numpy as np
import point_cloud_utils as pcu

from utils.utils_object import get_pt_idx_within_object
from utils.utils_misc import check_in_box, downsample_pc_voxel

from igibson import object_states


def check_if_camera_pose_valid(camera_pose, room_ins_info):
    # check if fall out of room
    if not (camera_pose[0] >= room_ins_info['xy_min'][0] \
        and camera_pose[0] <= room_ins_info['xy_max'][0] \
        and camera_pose[1] >= room_ins_info['xy_min'][1] \
        and camera_pose[1] <= room_ins_info['xy_max'][1]):
        return False
    
    # check if fall into any object
    if not check_in_box([camera_pose], room_ins_info["objects_bbox"], offset=0.):
        return False
    
    return True


def get_obs(renderer, camera_pose, view_direction, up_direction=[0, 0, 1], obs_types=[], savefig=False):

            renderer.set_camera(camera_pose, camera_pose + view_direction, up_direction)
            renderer.set_fov(90)
            
            transformation_matrix = np.eye(3)
            camera_pose_opengl = np.array([camera_pose[1], camera_pose[2], camera_pose[0]])
            
            output = {}
            output['camera_pose'] = camera_pose
            output['view_direction'] = view_direction
            output['up_direction'] = up_direction
            output['intrinsics'] = renderer.get_intrinsics()
            
            if 'rgb' in obs_types:
                frame_rgb = renderer.render(modes=("rgb"))[0][:, :, :3].astype(np.float32)
                output['rbg'] = frame_rgb
                if savefig:
                    img = Image.fromarray((frame_rgb * 255).astype(np.uint8), mode="RGB")
                    img.save('rgb.png')
                
            if 'seg' in obs_types:
                frame_seg = renderer.render(modes=("seg"))[0][:, :, 0].astype(np.float32)
                frame_seg = (frame_seg * 255.0).astype(np.uint8)
                output['seg'] = frame_seg
                if savefig:
                    img = Image.fromarray(frame_seg.astype(np.uint8), mode="P")
                    img.save('seg.png')
                
            if 'normal' in obs_types:
                frame_normal = renderer.render(modes=("normal"))[0][:, :, :3].astype(np.float32)
                output['normal'] = frame_normal
                if savefig:
                    img = Image.fromarray((frame_normal * 255).astype(np.uint8), mode="RGB")
                    img.save('normal.png')
                
            if '3d' in obs_types or 'depth' in obs_types or 'pc' in obs_types:
                frame_3d = renderer.render(modes=("3d"))[0][:, :, :3].astype(np.float32)
                
                if '3d' in obs_types:
                    output['3d'] = frame_3d
                
                if 'depth' in obs_types:
                    depth = -frame_3d[:, :, 2]
                    output['depth'] = depth
                    if savefig:
                        # depth = np.linalg.norm(frame_3d, axis=2)
                        # depth /= depth.max() + 1e-6
                        img = Image.fromarray((depth * 255).astype(np.uint8), mode='L')
                        img.save('depth.png')
                
                if 'pc' in obs_types:
                    pointcloud = frame_3d.dot(transformation_matrix).reshape(-1, 3) + camera_pose_opengl[None, :]
                    output['pc'] = pointcloud
                    if savefig:
                        fig = plt.figure()
                        ax = Axes3D(fig)
                        ax.scatter(pointcloud[:, 0], pointcloud[:, 2], pointcloud[:, 1], s=3)
                        plt.show()
                        plt.savefig('pc.png')
                    
            return output


def get_8_views(renderer, mode="rgb"):
    """
    :param mode: simulator rendering mode, 'rgb' or '3d'
    :return: List of sensor readings, normalized to [0.0, 1.0], ordered as [F, R, B, L, U, D] * n_cameras
    """

    # Cache the original fov and V to be restored later
    original_fov = renderer.vertical_fov
    original_V = np.copy(renderer.V)

    # Set fov to be 90 degrees
    renderer.set_fov(90)
    initial_V = original_V

    def render_cube():
        # Store 6 frames in 6 directions
        frames = []

        # Forward, backward, left, right
        r = np.array(
            [
                [
                    np.cos(-np.pi / 4),
                    0,
                    -np.sin(-np.pi / 4),
                    0,
                ],
                [0, 1, 0, 0],
                [np.sin(-np.pi / 4), 0, np.cos(-np.pi / 4), 0],
                [0, 0, 0, 1],
            ]
        )

        for i in range(8):
            frames.append(renderer.render(modes=(mode))[0])
            renderer.V = r.dot(renderer.V)

        # Up
        r_up = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

        renderer.V = r_up.dot(initial_V)
        frames.append(renderer.render(modes=(mode))[0])

        # Down
        r_down = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        renderer.V = r_down.dot(initial_V)
        frames.append(renderer.render(modes=(mode))[0])

        return frames

    frames = render_cube()

    # Restore original fov and V
    renderer.V = original_V
    renderer.set_fov(original_fov)

    return frames


def generate_data_lidar(simulator, camera_pose, view_direction, up_direction=np.array([0,0,1]), selected_frame_ind=None, n_views='4'):

    rgb_all = []
    lidar_all = []
    lidar_all_2 = []
    label_all = []

    # set camera
    simulator.renderer.set_camera(camera_pose, camera_pose + view_direction, up_direction)
    simulator.renderer.set_fov(90)
    
    if n_views == '4':    
        # Get observations (panorama RGB, 3D/Depth and semantic segmentation)
        pano_rgb = simulator.renderer.get_cube(mode="rgb", use_robot_camera=False)
        pano_3d = simulator.renderer.get_cube(mode="3d", use_robot_camera=False)
        pano_seg = simulator.renderer.get_cube(mode="seg", use_robot_camera=False)
        
        r3 = np.array(
            [[np.cos(-np.pi / 2), 0, -np.sin(-np.pi / 2)], [0, 1, 0], [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]]
        )
        transformation_matrix = np.eye(3)

        for i in range(4):
            camera_pose_opengl = np.array([camera_pose[1], camera_pose[2], camera_pose[0]])
            lidar_all.append(pano_3d[i][:, :, :3].dot(transformation_matrix).reshape(-1, 3) + camera_pose_opengl[None, :]) ##[x_samples, y_samples] )#- delta_pos[None, :])
            rgb_all.append(pano_rgb[i][:, :, :3].reshape(-1, 3))#[x_samples, y_samples])
            label_all.append(pano_seg[i][:, :, 0] * 255.0)#[x_samples, y_samples] * 255.0)
            lidar_all_2.append(
                pano_3d[i][:, :, :3].dot(transformation_matrix).reshape(-1, 3)#[x_samples, y_samples] * 0.9 #- delta_pos[None, :]
            )            
            transformation_matrix = r3.dot(transformation_matrix)
    elif n_views == '8':
        # Get observations (panorama RGB, 3D/Depth and semantic segmentation)
        pano_rgb = get_8_views(simulator.renderer, mode="rgb")
        pano_3d = get_8_views(simulator.renderer, mode="3d")
        pano_seg = get_8_views(simulator.renderer, mode="seg")
        
        r3 = np.array(
            [[np.cos(-np.pi / 4), 0, -np.sin(-np.pi / 4)], [0, 1, 0], [np.sin(-np.pi / 4), 0, np.cos(-np.pi / 4)]]
        )
        transformation_matrix = np.eye(3)

        for i in range(8):
            camera_pose_opengl = np.array([camera_pose[1], camera_pose[2], camera_pose[0]])
            lidar_all.append(pano_3d[i][:, :, :3].dot(transformation_matrix).reshape(-1, 3) + camera_pose_opengl[None, :]) ##[x_samples, y_samples] )#- delta_pos[None, :])
            rgb_all.append(pano_rgb[i][:, :, :3].reshape(-1, 3))#[x_samples, y_samples])
            label_all.append(pano_seg[i][:, :, 0] * 255.0)#[x_samples, y_samples] * 255.0)
            lidar_all_2.append(
                pano_3d[i][:, :, :3].dot(transformation_matrix).reshape(-1, 3)#[x_samples, y_samples] * 0.9 #- delta_pos[None, :]
            )            
            transformation_matrix = r3.dot(transformation_matrix)
    else:
        raise NotImplementedError
    
    if selected_frame_ind is not None:
        lidar_all = lidar_all[selected_frame_ind].astype(np.float32)
        lidar_all_2 = lidar_all_2[selected_frame_ind].astype(np.float32)
        rgb_all = rgb_all[selected_frame_ind].astype(np.float32)
        label_all = label_all[selected_frame_ind].astype(np.int32)
    else:
        lidar_all = np.concatenate(lidar_all, 0).astype(np.float32)
        lidar_all_2 = np.concatenate(lidar_all_2, 0).astype(np.float32)
        rgb_all = np.concatenate(rgb_all, 0).astype(np.float32)
        label_all = np.concatenate(label_all, 0).astype(np.int32)
    
    direction = lidar_all - lidar_all_2
    direction = direction / (np.linalg.norm(direction, axis=1)[:, None] + 1e-5)

    return lidar_all, direction, rgb_all, label_all


def generate_scene_pointcloud(simulator, room_info_ins, step_size=1.5, stride=0.5, sizeof_voxel_pc=0.025, offset_room_crop=0., n_views='4'):
    
    # sample pointcloud
    new_pts = []
    new_color = []
    new_label = []
    camera_pose_list = []
    
    # get pc blocks
    grid_x = int(np.ceil(float(room_info_ins['xy_max'][0] - room_info_ins['xy_min'][0] - step_size) / stride) + 1)
    grid_y = int(np.ceil(float(room_info_ins['xy_max'][1] - room_info_ins['xy_min'][1] - step_size) / stride) + 1)
    if grid_x <= 0:
        grid_x = 1
    if grid_y <= 0:
        grid_y = 1
    
    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            for index_z in np.linspace(0., 2.5, num=3):
                s_x = room_info_ins['xy_min'][0] + index_x * stride
                e_x = min(s_x + step_size, room_info_ins['xy_max'][0])
                s_x = e_x - step_size
                c_x = (s_x+e_x) / 2.
                s_y = room_info_ins['xy_min'][1] + index_y * stride
                e_y = min(s_y + step_size, room_info_ins['xy_max'][1])
                s_y = e_y - step_size
                c_y = (s_y+e_y) / 2.
                c_z = index_z
                
                camera_pose_xyz = np.array([c_x, c_y, c_z])
                
                # check if fall into any object
                if not check_in_box([camera_pose_xyz], room_info_ins["objects_bbox"], offset=0.):
                    break
                
                camera_pose_list.append(camera_pose_xyz)
                
                # generate scene pointcloud
                view_direction = np.array([-1, 0, 0])
                pts_temp, direction_temp, color_temp, label_temp = generate_data_lidar(simulator, camera_pose_xyz, view_direction, n_views=n_views)
                color_temp = color_temp.reshape(-1, 3)
                label_temp = label_temp.reshape(-1)
                
                # crop by room size
                selected_idx = np.where(
                        (pts_temp[:, 2] >= room_info_ins['xy_min'][0] - offset_room_crop) & 
                        (pts_temp[:, 2] <= room_info_ins['xy_max'][0] + offset_room_crop) & 
                        (pts_temp[:, 0] >= room_info_ins['xy_min'][1] - offset_room_crop) & 
                        (pts_temp[:, 0] <= room_info_ins['xy_max'][1] + offset_room_crop)
                    )[0]
                new_pts_temp = pts_temp[selected_idx]
                new_color_temp = color_temp[selected_idx]
                new_label_temp = label_temp[selected_idx]
                
                # remove floor wall ceilings  # 0: floor, 1: wall, 2:ceiling
                selected_idx = np.where(
                    np.isin(new_label_temp, [0, 1, 2]) 
                )[0]
                mask = np.full((len(new_pts_temp), ), True)
                mask[selected_idx] = False
                new_pts_temp = new_pts_temp[mask]
                new_color_temp = new_color_temp[mask]
                new_label_temp = new_label_temp[mask]   
                
                # check if pts
                if len(new_pts_temp) <= 10:
                    continue 
                
                # # downsample the pointcloud
                # new_pts_temp, new_color_temp, new_label_temp = downsample_pc_voxel(new_pts_temp, sizeof_voxel_pc, color=new_color_temp, label=new_label_temp)          
                
                new_pts.append(new_pts_temp)
                new_color.append(new_color_temp)
                new_label.append(new_label_temp)
    
    camera_pose_list = np.stack(camera_pose_list)
    new_pts = np.concatenate(new_pts, axis=0)        
    new_color = np.concatenate(new_color, axis=0)
    new_label = np.concatenate(new_label, axis=0)
        
    # downsample the pointcloud
    new_pts, new_color, new_label = downsample_pc_voxel(new_pts, sizeof_voxel_pc, color=new_color, label=new_label)
    
    # remove collision
    mask = np.full((len(new_pts), ), True)
    new_pts_opengl = np.stack([new_pts[:, 2], new_pts[:, 0], new_pts[:, 1]], axis=-1) # transform into opengl frame
    
    for obj in simulator.scene.objects_by_state[object_states.Open]:
        if obj.name in simulator.scene.body_collision_set or obj.name in simulator.scene.link_collision_set:
            print(obj.name)
            selected_idx = get_pt_idx_within_object(new_pts_opengl, obj)
            mask[selected_idx] = False
    
    new_pts = new_pts[mask]
    new_color = new_color[mask]
    new_label = new_label[mask]
    
    # deduplicate pointcloud
    # idx_i is an array of indices such that p_dedup = p[idx_i]
    # idx_j is an array of indices such that p = p_dedup[idx_j]
    new_pts, idx_i, idx_j  = pcu.deduplicate_point_cloud(new_pts, 1e-7)
    new_color = new_color[idx_i]
    new_label = new_label[idx_i]
    
    return new_pts, new_color, new_label, camera_pose_list


def generate_partial_pointcloud(hit_pos, simulator, room_info_ins, extent_factor=0.75, sizeof_voxel_pc=0.015, offset_room_crop=0., offset_obj_crop=0.75):
    
    # sample camera pose
    camera_pose_list = []
    selected_extent_vector_list = []
    
    extent_vector_list = [
        (np.array([1, 0, 0]), 0, True), 
        (np.array([1, -1, 0]), 1, True),       
        (np.array([0, -1, 0]), 2, True), 
        (np.array([-1, -1, 0]), 3, True), 
        (np.array([-1, 0, 0]), 4, True),
        (np.array([-1, 1, 0]), 5, True), 
        (np.array([0, 1, 0]), 6, True), 
        (np.array([1, 1, 0]), 7, True),
    ]
    height_list = [0.5, 1.0, 1.5, 2.0]
    
    # TODO: sample different height
    for idx_extent_vector, (extent_vector, frame_ind, if_use) in enumerate(extent_vector_list):
        if not if_use:
            continue 
        
        camera_pose_count_before = len(camera_pose_list)
        
        for height in height_list:
            # sample current camera pose
            camera_pose = hit_pos + extent_factor * extent_vector
            camera_pose[2] = height
            if check_if_camera_pose_valid(camera_pose, room_info_ins):
                camera_pose_list.append((camera_pose, frame_ind))
                # p.addUserDebugLine(hit_pos-0.025, hit_pos+0.025, lineWidth=4, lineColorRGB=[0., 1., 0.])
                # p.addUserDebugLine(camera_pose-0.025, camera_pose+0.025, lineWidth=4, lineColorRGB=[1., 0., 0.])
        
        camera_pose_count_after = len(camera_pose_list)
        
        if camera_pose_count_after > camera_pose_count_before:
            selected_extent_vector_list.append(extent_vector)
    
    # check if any valid camera pose
    if len(camera_pose_list) <= 0:
        return None, None, None, None
    
    # check if enough sampled angle
    # if len(selected_extent_vector_list) < 3:
    if len(selected_extent_vector_list) < 1:
        return None, None, None, None
        
    # sample pointcloud
    new_pts = []
    new_color = []
    new_label = []
    
    for camera_pose, frame_ind in camera_pose_list:
        view_direction = np.array([-1, 0, 0])
        pts_temp, direction_temp, color_temp, label_temp = generate_data_lidar(simulator, camera_pose, view_direction, n_views='8', selected_frame_ind=frame_ind)
        color_temp = color_temp.reshape(-1, 3)
        label_temp = label_temp.reshape(-1)
        
        # crop by room size
        selected_idx = np.where(
                (pts_temp[:, 2] >= room_info_ins['xy_min'][0] - offset_room_crop) & 
                (pts_temp[:, 2] <= room_info_ins['xy_max'][0] + offset_room_crop) & 
                (pts_temp[:, 0] >= room_info_ins['xy_min'][1] - offset_room_crop) & 
                (pts_temp[:, 0] <= room_info_ins['xy_max'][1] + offset_room_crop)
            )[0]
        new_pts_temp = pts_temp[selected_idx]
        new_color_temp = color_temp[selected_idx]
        new_label_temp = label_temp[selected_idx]
        
        # crop by hit_pos
        selected_idx = np.where(
                (pts_temp[:, 2] >= hit_pos[0] - offset_obj_crop) & 
                (pts_temp[:, 2] <= hit_pos[0] + offset_obj_crop) & 
                (pts_temp[:, 0] >= hit_pos[1] - offset_obj_crop) & 
                (pts_temp[:, 0] <= hit_pos[1] + offset_obj_crop)
            )[0]
        new_pts_temp = pts_temp[selected_idx]
        new_color_temp = color_temp[selected_idx]
        new_label_temp = label_temp[selected_idx]
        
        # remove floor wall ceilings  # 0: floor, 1: wall, 2:ceiling
        selected_idx = np.where(
            np.isin(new_label_temp, [0, 1, 2]) 
        )[0]
        mask = np.full((len(new_pts_temp), ), True)
        mask[selected_idx] = False
        new_pts_temp = new_pts_temp[mask]
        new_color_temp = new_color_temp[mask]
        new_label_temp = new_label_temp[mask]   
        
        # check if pts
        if len(new_pts_temp) <= 10:
            continue 
        
        # # downsample the pointcloud
        # new_pts_temp, new_color_temp, new_label_temp = downsample_pc_voxel(new_pts_temp, sizeof_voxel_pc, color=new_color_temp, label=new_label_temp)          
        
        new_pts.append(new_pts_temp)
        new_color.append(new_color_temp)
        new_label.append(new_label_temp)
    
    
    # post-processing
    camera_pose_list = np.stack(camera_pose_list)
    new_pts = np.concatenate(new_pts, axis=0)        
    new_color = np.concatenate(new_color, axis=0)
    new_label = np.concatenate(new_label, axis=0)
        
    # downsample the pointcloud
    new_pts, new_color, new_label = downsample_pc_voxel(new_pts, sizeof_voxel_pc, color=new_color, label=new_label)
    
    # remove collision 
    mask = np.full((len(new_pts), ), True)
    new_pts_opengl = np.stack([new_pts[:, 2], new_pts[:, 0], new_pts[:, 1]], axis=-1) # transform into opengl frame
    
    for obj in simulator.scene.objects_by_state[object_states.Open]:
        if obj.name in simulator.scene.body_collision_set or obj.name in simulator.scene.link_collision_set:
            selected_idx = get_pt_idx_within_object(new_pts_opengl, obj)
            mask[selected_idx] = False
    
    new_pts = new_pts[mask]
    new_color = new_color[mask]
    new_label = new_label[mask]
    
    return new_pts, new_color, new_label, camera_pose_list