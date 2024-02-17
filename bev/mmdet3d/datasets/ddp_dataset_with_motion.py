import tempfile
from os import path as osp

import mmcv
import numpy as np
import pyquaternion
import torch
from mmdet.datasets import DATASETS
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from .custom_3d import Custom3DDataset
from .data_utils.geometry import invert_matrix_egopose_numpy, mat2pose_vec
from ..core.bbox import LiDARInstance3DBoxes
from nuscenes.nuscenes import NuScenes
import cv2
from nuscenes.utils.data_classes import Box


NUM_COLORS = 25
# 生成颜色表
np.random.seed(0)  # 设置随机种子以保证颜色的一致性
COLORS = np.random.randint(0, 256, size=(NUM_COLORS, 3)).tolist()
# 确保值0对应的颜色是白色
COLORS[0] = [255, 255, 255]
# 将颜色列表转换为元组
COLORS = [tuple(color) for color in COLORS]

@DATASETS.register_module()
class DDPDataset_WITH_MOTION(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        dataset_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    AttrMapping = {
        "cycle.with_rider": 0,
        "cycle.without_rider": 1,
        "pedestrian.moving": 2,
        "pedestrian.standing": 3,
        "pedestrian.sitting_lying_down": 4,
        "vehicle.moving": 5,
        "vehicle.parked": 6,
        "vehicle.stopped": 7,
    }
    AttrMapping_rev = [
        "cycle.with_rider",
        "cycle.without_rider",
        "pedestrian.moving",
        "pedestrian.standing",
        "pedestrian.sitting_lying_down",
        "vehicle.moving",
        "vehicle.parked",
        "vehicle.stopped",
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )

    def __init__(
            self,
            ann_file,
            pipeline=None,
            dataset_root=None,
            object_classes=None,
            map_classes=None,
            load_interval=1,
            with_velocity=True,
            modality=None,
            version="mini",
            box_type_3d="LiDAR",
            filter_empty_gt=True,
            test_mode=False,
            eval_version="detection_cvpr_2019",
            use_valid_flag=False,
            receptive_field=1,
            future_frames=0,
            motion_pred_grid_conf=None,
            filter_invalid_sample=True,
            filter_invisible_vehicles=True,
    ) -> None:
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        self.map_classes = map_classes
        self.nusc = NuScenes(version='v1.0-{}'.format(version), dataroot=dataset_root, verbose=False)
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory

        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.receptive_field = receptive_field
        self.n_future = future_frames
        self.sequence_length = receptive_field + future_frames
        self.filter_invalid_sample = filter_invalid_sample
        self.data_infos.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        ### BELOW IS FOR MOTION PREDICTION, CURRENTLY DON'T CARE ###
        self.filter_invisible_vehicles=filter_invisible_vehicles
        self.motion_pred_grid_conf = motion_pred_grid_conf
        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.motion_pred_grid_conf.xbound, self.motion_pred_grid_conf.ybound, self.motion_pred_grid_conf.zbound
        )  # (0.5, 0.5, 0.5), (row[0]+row[2])/2 (200, 200, 1)
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.motion_pred_grid_conf.xbound[1], self.motion_pred_grid_conf.ybound[1])  # (50,50)
    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_temporal_indices(self, index):
        current_scene_token = self.data_infos[index]['scene_token']

        # generate the past
        previous_indices = []
        for t in range(- self.receptive_field + 1, 0):
            index_t = index + t
            if index_t >= 0 and self.data_infos[index_t]['scene_token'] == current_scene_token:
                previous_indices.append(index_t)
            else:
                previous_indices.append(-1)  # for invalid indices

        # generate the future
        future_indices = []
        for t in range(1, self.n_future + 1):
            index_t = index + t
            if index_t < len(self.data_infos) and self.data_infos[index_t]['scene_token'] == current_scene_token:
                future_indices.append(index_t)
            else:
                future_indices.append(-1)

        return previous_indices, future_indices

    @staticmethod
    def get_egopose_from_info(info):
        # ego2global transformation (lidar_ego)
        e2g_trans_matrix = np.zeros((4, 4), dtype=np.float32)
        e2g_rot = info['ego2global_rotation']
        e2g_trans = info['ego2global_translation']
        e2g_trans_matrix[:3, :3] = pyquaternion.Quaternion(
            e2g_rot).rotation_matrix
        e2g_trans_matrix[:3, 3] = np.array(e2g_trans)
        e2g_trans_matrix[3, 3] = 1.0

        return e2g_trans_matrix

    def get_egomotions(self, indices):
        # get ego_motion for each frame
        future_egomotions = []
        for index in indices:
            cur_info = self.data_infos[index]
            ego_motion = np.eye(4, dtype=np.float32)
            next_frame = index + 1

            # 如何处理 invalid frame
            if index != -1 and next_frame < len(self.data_infos) and self.data_infos[next_frame]['scene_token'] == \
                    cur_info['scene_token']:
                next_info = self.data_infos[next_frame]
                # get ego2global transformation matrices
                cur_egopose = self.get_egopose_from_info(cur_info)
                next_egopose = self.get_egopose_from_info(next_info)

                # trans from cur to next
                ego_motion = invert_matrix_egopose_numpy(
                    next_egopose).dot(cur_egopose)  # for ego, from current to next frame
                ego_motion[3, :3] = 0.0
                ego_motion[3, 3] = 1.0

            # transformation between adjacent frames
            # if index == -1 --> then ego_motion is identity
            ego_motion = torch.Tensor(ego_motion).float()
            ego_motion = mat2pose_vec(ego_motion)  # transform from a 4x4 matrix to 6 DoF vector
            future_egomotions.append(ego_motion)

        return torch.stack(future_egomotions, dim=0)

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        # when the labels for this frame window are not complete, skip the sample
        if self.filter_invalid_sample and input_dict['has_invalid_frame'] is True:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        if self.filter_empty_gt and (example is None or
                                     ~(example['gt_labels_3d']._data != -1).any()):
            return None
        if "points" not in example:
            example["points"] = []
        return example

    def record_instance(self, rec, instance_map):
        """
        Record information about each visible instance in the sequence and assign a unique ID to it.
        """
        translation, rotation = self._get_top_lidar_pose(rec)
        self.egopose_list.append([translation, rotation])

        for annotation_token in rec['anns']:

            annotation = self.nusc.get('sample_annotation', annotation_token)

            # NuScenes filter
            # Filter out all non vehicle instances
            if 'vehicle' not in annotation['category_name']:
                continue
            # Filter out invisible vehicles
            if self.filter_invisible_vehicles and int(annotation['visibility_token']) == 1 and \
                    annotation['instance_token'] not in self.visible_instance_set:
                continue
            # Filter out vehicles that have not been seen in the past
            if self.counter >= self.receptive_field and annotation[
                'instance_token'] not in self.visible_instance_set:
                continue
            self.visible_instance_set.add(annotation['instance_token'])

            if annotation['instance_token'] not in instance_map:
                instance_map[annotation['instance_token']] = len(instance_map) + 1
            instance_id = instance_map[annotation['instance_token']]

            instance_attribute = int(annotation['visibility_token'])

            if annotation['instance_token'] not in self.instance_dict:
                # For the first occurrence of an instance
                self.instance_dict[annotation['instance_token']] = {
                    'timestep': [self.counter],
                    'translation': [annotation['translation']],
                    'rotation': [annotation['rotation']],
                    'size': annotation['size'],
                    'instance_id': instance_id,
                    'attribute_label': [instance_attribute],
                }
            else:
                # For the instance that have appeared before
                self.instance_dict[annotation['instance_token']]['timestep'].append(self.counter)
                self.instance_dict[annotation['instance_token']]['translation'].append(annotation['translation'])
                self.instance_dict[annotation['instance_token']]['rotation'].append(annotation['rotation'])
                self.instance_dict[annotation['instance_token']]['attribute_label'].append(instance_attribute)

        return instance_map

    def refine_instance_poly(self, instance):
        """
        Fix the missing frames and disturbances of ground truth caused by noise.
        """
        pointer = 1
        for i in range(instance['timestep'][0] + 1, self.sequence_length):
            # Fill in the missing frames
            if i not in instance['timestep']:
                instance['timestep'].insert(pointer, i)
                instance['translation'].insert(pointer, instance['translation'][pointer - 1])
                instance['rotation'].insert(pointer, instance['rotation'][pointer - 1])
                instance['attribute_label'].insert(pointer, instance['attribute_label'][pointer - 1])
                pointer += 1
                continue

            # Eliminate observation disturbances
            if self._check_consistency(instance['translation'][pointer], instance['translation'][pointer - 1]):
                instance['translation'][pointer] = instance['translation'][pointer - 1]
                instance['rotation'][pointer] = instance['rotation'][pointer - 1]
                instance['attribute_label'][pointer] = instance['attribute_label'][pointer - 1]
            pointer += 1

        return instance
    @staticmethod
    def _check_consistency(translation, prev_translation, threshold=1.0):
        """
        Check for significant displacement of the instance adjacent moments.
        """
        x, y = translation[:2]
        prev_x, prev_y = prev_translation[:2]

        if abs(x - prev_x) > threshold or abs(y - prev_y) > threshold:
            return False
        return True

    @staticmethod
    def generate_flow(flow, instance_img, instance, instance_id):
        """
        Generate ground truth for the flow of each instance based on instance segmentation.
        """
        _, h, w = instance_img.shape
        x, y = torch.meshgrid(torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float))
        grid = torch.stack((x, y), dim=0)  # (2,h,w)

        # flow with shape (seq,2,h,w)

        # Set the first frame
        instance_mask = (instance_img[0] == instance_id)
        # here x, y should be changed because of the loading method of matrix
        flow[0, 1, instance_mask] = grid[0, instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
        flow[0, 0, instance_mask] = grid[1, instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]
        # beyond set the vector from each instance grid to the center of this instance. The center of each instance is set by mean
        for i, timestep in enumerate(instance['timestep']):
            if i == 0:
                continue

            instance_mask = (instance_img[timestep] == instance_id)
            prev_instance_mask = (instance_img[timestep - 1] == instance_id)
            if instance_mask.sum() == 0 or prev_instance_mask.sum() == 0:
                continue

            # Centripetal backward flow is defined as displacement vector from each foreground pixel at time t to the object center of the associated instance identity at time t−1
            flow[timestep, 1, instance_mask] = grid[0, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[
                0, instance_mask]
            flow[timestep, 0, instance_mask] = grid[1, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[
                1, instance_mask]

        return flow

    def get_flow_label(self, instance_img, instance_map, ignore_index=255):
        """
        Generate the global map of the flow ground truth.
        """
        seq_len, h, w = instance_img.shape
        flow = ignore_index * torch.ones(seq_len, 2, h, w)

        for token, instance in self.instance_dict.items():  # iterate through each instance
            flow = self.generate_flow(flow, instance_img, instance, instance_map[token])
        return flow

    def _get_poly_region_in_image(self, instance_annotation, present_egopose):
        """
        Obtain the bounding box polygon of the instance.
        """
        present_ego_translation, present_ego_rotation = present_egopose

        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(present_ego_translation)
        box.rotate(present_ego_rotation)
        pts = box.bottom_corners()[:2].T

        if self.motion_pred_grid_conf.xbound[0] <= pts.min(axis=0)[0] and pts.max(axis=0)[0] <= self.motion_pred_grid_conf.xbound[1] and \
                self.motion_pred_grid_conf.ybound[0] <= pts.min(axis=0)[1] and pts.max(axis=0)[1] <= self.motion_pred_grid_conf.ybound[1]:
            pts = np.round(
                (pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(
                np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            z = box.bottom_corners()[2, 0]
            return pts, z
        else:
            return None, None
    def _get_top_lidar_pose(self, rec):
        """
        Obtain the vehicle attitude at the current moment.
        """
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot
    def get_label(self):
        """
        Generate labels for semantic segmentation, instance segmentation, z position, attribute from the raw data of nuScenes.
        """
        timestep = self.counter
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        #z_position = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        attribute_label = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

        for instance_token, instance_annotation in self.instance_dict.items():
            if timestep not in instance_annotation['timestep']:
                continue
            pointer = instance_annotation['timestep'].index(timestep)
            annotation = {
                'translation': instance_annotation['translation'][pointer],
                'rotation': instance_annotation['rotation'][pointer],
                'size': instance_annotation['size'],
            }
            poly_region, _ = self._get_poly_region_in_image(annotation,
                                                            self.egopose_list[self.receptive_field - 1])
            if isinstance(poly_region, np.ndarray):
                if self.counter >= self.receptive_field and instance_token not in self.visible_instance_set:
                    # 如果超出receptive_field，且这个token不在visible_instance_set里面，就不要了
                    continue
                self.visible_instance_set.add(instance_token)

                cv2.fillPoly(instance, [poly_region], instance_annotation['instance_id'])
                cv2.fillPoly(segmentation, [poly_region], 1.0)
                cv2.fillPoly(attribute_label, [poly_region],
                             instance_annotation['attribute_label'][pointer])  # visible or not

        segmentation = torch.from_numpy(segmentation).long().unsqueeze(0).unsqueeze(0)
        instance = torch.from_numpy(instance).long().unsqueeze(0)
        #z_position = torch.from_numpy(z_position).float().unsqueeze(0).unsqueeze(0)
        attribute_label = torch.from_numpy(attribute_label).long().unsqueeze(0).unsqueeze(0)

        return segmentation, instance, attribute_label
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info["location"],
        )
        # get the previous and future indices
        prev_indices, future_indices = self.get_temporal_indices(index)
        # ego motions are needed for all frames
        all_frames = prev_indices + [index] + future_indices
        # [num_seq, 6 DoF]
        future_egomotions = self.get_egomotions(all_frames)
        data['future_egomotions'] = future_egomotions
        # whether invalid frame is present
        has_invalid_frame = -1 in all_frames
        data['has_invalid_frame'] = has_invalid_frame
        data['img_is_valid'] = np.array(all_frames) >= 0
        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for key, camera_info in info["cams"].items():
                data["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                        camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)


        label_frames = [index] + future_indices


        # now we loading the instance trajectories
        # The visible instance must have high visibility in the time receptive field
        # Record all valid instance
        self.instance_dict = {}
        instance_map = {} # token to id
        self.egopose_list=[]
        self.visible_instance_set = set()

        data['sample_token']=[]
        data["segmentation"]=[]
        data['instance']=[]
        data['attribute']=[]
        print(f"label frames: {label_frames}")
        for self.counter,index_t in enumerate(label_frames):
            rec = self.nusc.get('sample', self.data_infos[index_t]['token'])
            instance_map = self.record_instance(rec, instance_map)
            data['sample_token'].append(rec['token'])


        for token in self.instance_dict.keys():
            self.instance_dict[token] = self.refine_instance_poly(self.instance_dict[token])

        # The visible instance must have high visibility in the time receptive field
        self.visible_instance_set = set()
        # Generate instance ground truth
        for self.counter in range(self.sequence_length):
            segmentation, instance, attribute_label = self.get_label()
            data['segmentation'].append(segmentation) #(T,1,bev_h,bev_w)
            data['instance'].append(instance) #(T,bev_h,bev_w)
            data['attribute'].append(attribute_label) # (T,1,bev_h,bev_w)

        for key, value in data.items():
            if key in ['segmentation', 'instance', 'attribute']:
                data[key] = torch.cat(value, dim=0)


        for i in range(data['segmentation'].shape[0]):
            seg = data['segmentation'][i]
            ins = data['instance'][i]
            seg_img = np.uint8(seg[0] * 255)  # 转换为二值图像
            seg_img = cv2.bitwise_not(seg_img)  # 反转颜色（使0变为白色，1变为黑色）
            # 保存segmentation map
            cv2.imwrite(f'./figures/test/segmentation_{i}.jpg', seg_img)

            # 处理instance map
            # 为每个不同的值分配一个随机颜色
            instance_img = np.zeros((200, 200, 3), dtype=np.uint8)  # 创建一个空彩色图像
            unique_values = np.unique(ins)
            for value in unique_values:
                if value != 0:
                    instance_img[ins == value] = COLORS[value]
                else:  # 如果值是0，分配白色
                    instance_img[ins == value] = (255, 255, 255)
            # 保存instance map
            cv2.imwrite(f'./figures/test/instance_{i}.jpg', instance_img)
        # Generate centripetal backward flow ground truth from the instance ground truth
        # need instance_image and token_to_id dict
        data['flow'] = self.get_flow_label(data['instance'], instance_map,
                                           ignore_index=255)

        return data

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.eval_detection_configs,
                self.eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
            self,
            result_path,
            logger=None,
            metric="bbox",
            result_name="pts_bbox",
    ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import DetectionEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version, dataroot=self.dataset_root, verbose=False)
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        nusc_eval = DetectionEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
        )
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
        detail = dict()
        for name in self.CLASSES:
            for k, v in metrics["label_aps"][name].items():
                val = float("{:.4f}".format(v))
                detail["object/{}_ap_dist_{}".format(name, k)] = val
            for k, v in metrics["label_tp_errors"][name].items():
                val = float("{:.4f}".format(v))
                detail["object/{}_{}".format(name, k)] = val
            for k, v in metrics["tp_errors"].items():
                val = float("{:.4f}".format(v))
                detail["object/{}".format(self.ErrNameMapping[k])] = val

        detail["object/nds"] = metrics["nd_score"]
        detail["object/map"] = metrics["mean_ap"]
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate_map(self, results):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["masks_bev"]
            label = result["gt_masks_bev"]

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics

    def evaluate(
            self,
            results,
            metric="bbox",
            jsonfile_prefix=None,
            result_names=["pts_bbox"],
            **kwargs,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        metrics = {}

        if "masks_bev" in results[0]:
            metrics.update(self.evaluate_map(results))

        if "boxes_3d" in results[0]:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                for name in result_names:
                    print("Evaluating bboxes of {}".format(name))
                    ret_dict = self._evaluate_single(result_files[name])
                metrics.update(ret_dict)
            elif isinstance(result_files, str):
                metrics.update(self._evaluate_single(result_files))

            if tmp_dir is not None:
                tmp_dir.cleanup()

        return metrics


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
        info, boxes, classes, eval_configs, eval_version="detection_cvpr_2019"
):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs : Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))

        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list
def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension
