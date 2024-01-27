import cv2
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES

from ..data_utils.instance import convert_instance_mask_to_center_and_offset_label_with_warper
from ..data_utils.warper import FeatureWarper
from ...visualize import Visualizer


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
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                  for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


@PIPELINES.register_module()
class ConvertMotionLabels(object):
    def __init__(self, grid_conf, ignore_index=255, only_vehicle=True, filter_invisible=True, receptive_field=1):
        self.grid_conf = grid_conf
        # torch.tensor
        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
        )
        # convert numpy
        self.bev_resolution = self.bev_resolution.numpy()
        self.bev_start_position = self.bev_start_position.numpy()
        self.bev_dimension = self.bev_dimension.numpy()
        self.spatial_extent = (grid_conf['xbound'][1], grid_conf['ybound'][1])
        self.ignore_index = ignore_index
        self.only_vehicle = only_vehicle
        self.filter_invisible = filter_invisible

        nusc_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        vehicle_classes = ['car', 'bus', 'construction_vehicle',
                           'bicycle', 'motorcycle', 'truck', 'trailer']

        self.vehicle_cls_ids = np.array([nusc_classes.index(
            cls_name) for cls_name in vehicle_classes])

        self.warper = FeatureWarper(grid_conf=grid_conf)
        self.receptive_field = receptive_field
        self.visualizer = Visualizer(out_dir="./figures/test", coordinate_system="ego")

    def __call__(self, results):
        # annotation_token ==> instance_id
        instance_map = {}

        # convert LiDAR bounding boxes to motion labels
        num_frame = len(results['gt_bboxes_3d'])
        all_gt_bboxes_3d = results['gt_bboxes_3d']
        all_gt_labels_3d = results['gt_labels_3d']
        all_instance_tokens = results['instance_tokens']
        all_vis_tokens = results['gt_vis_tokens']
        # 4x4 transformation matrix (if exist)
        bev_transform = results.get('aug_transform', None)

        segmentations = []
        instances = []

        # 对于 invalid frame: 所有 label 均为 255 -> 白色
        # 对于 valid frame: seg & instance 背景是 0(黑色），其它背景为255（白色）

        for frame_index in range(num_frame):
            gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d[frame_index], all_gt_labels_3d[frame_index]
            instance_tokens = all_instance_tokens[frame_index]
            vis_tokens = all_vis_tokens[frame_index]

            if gt_bboxes_3d is None:
                # for invalid samples
                segmentation = np.ones(
                    (self.bev_dimension[1], self.bev_dimension[0])) * self.ignore_index
                instance = np.ones(
                    (self.bev_dimension[1], self.bev_dimension[0])) * self.ignore_index
            else:
                # for valid samples
                segmentation = np.zeros(
                    (self.bev_dimension[1], self.bev_dimension[0]))
                instance = np.zeros(
                    (self.bev_dimension[1], self.bev_dimension[0]))

                if self.only_vehicle:
                    vehicle_mask = np.isin(gt_labels_3d, self.vehicle_cls_ids)
                    gt_bboxes_3d = gt_bboxes_3d[vehicle_mask]
                    gt_labels_3d = gt_labels_3d[vehicle_mask]
                    instance_tokens = instance_tokens[vehicle_mask]
                    vis_tokens = vis_tokens[vehicle_mask]

                if self.filter_invisible:
                    visible_mask = (vis_tokens != 1)
                    gt_bboxes_3d = gt_bboxes_3d[visible_mask]
                    gt_labels_3d = gt_labels_3d[visible_mask]
                    instance_tokens = instance_tokens[visible_mask]

                # valid sample and has objects
                if len(gt_bboxes_3d.tensor) > 0:
                    bbox_corners = gt_bboxes_3d.corners[:, [
                                                               0, 3, 7, 4], :2].numpy()
                    bbox_corners = np.round(
                        (bbox_corners - self.bev_start_position[:2] + self.bev_resolution[
                                                                      :2] / 2.0) / self.bev_resolution[:2]).astype(
                        np.int32)

                    for index, instance_token in enumerate(instance_tokens):
                        if instance_token not in instance_map:
                            instance_map[instance_token] = len(
                                instance_map) + 1

                        # instance_id start from 1
                        instance_id = instance_map[instance_token]
                        poly_region = bbox_corners[index]
                        cv2.fillPoly(segmentation, [poly_region], 1.0)
                        cv2.fillPoly(instance, [poly_region], instance_id)

            segmentations.append(segmentation)
            instances.append(instance)

        # segmentation = 1 where objects are located
        segmentations = torch.from_numpy(
            np.stack(segmentations, axis=0)).long()
        instances = torch.from_numpy(np.stack(instances, axis=0)).long()
        # generate heatmap & offset from segmentation & instance
        # shape of future_egomotion (num_frames,6)
        future_egomotions = results['future_egomotions'][- num_frame:]
        instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label_with_warper(
            instance_img=instances,
            future_egomotion=future_egomotions,
            num_instances=len(instance_map),
            ignore_index=self.ignore_index,
            subtract_egomotion=True,
            warper=self.warper,
            bev_transform=bev_transform,
        )

        invalid_mask = (segmentations[:, 0, 0] == self.ignore_index)
        instance_centerness[invalid_mask] = self.ignore_index

        # only keep detection labels for the current frame
        results['gt_bboxes_3d'] = all_gt_bboxes_3d[0]
        results['gt_labels_3d'] = all_gt_labels_3d[0]
        results['instance_tokens'] = all_instance_tokens[0]
        results['gt_valid_flag'] = results['gt_valid_flag'][0]
        results.update({
            'motion_segmentation': segmentations,  # (num_frames, bev_h,bev_w)
            'motion_instance': instances,  # (num_frames, bev_h,bev_w)  -> ground truth
            'instance_centerness': instance_centerness,  # (num_frames, 1, bev_h, bev_w)
            'instance_offset': instance_offset,  # (num_frames, 2, bev_h, bev_w)
            'instance_flow': instance_flow,  # (num_frames, 2, bev_h, bev_w)
        })
        # self.visualization(results,save_path='./figures/test')
        return results

    def visualization(self, results, save_path=None):
        motion_labels, _ = self.prepare_future_labels(results)
        self.visualizer.visualize_motion_gif(labels=motion_labels)
        self.visualizer.visualize_gt_motion(motion_labels=motion_labels, save_path=save_path)

    def prepare_future_labels(self, results):
        labels = {}
        future_distribution_inputs = []

        segmentation_labels = results['motion_segmentation'].unsqueeze(0)
        instance_center_labels = results['instance_centerness'].unsqueeze(0)
        instance_offset_labels = results['instance_offset'].unsqueeze(0)
        instance_flow_labels = results['instance_flow'].unsqueeze(0)
        gt_instance = results['motion_instance'].unsqueeze(0)
        future_egomotion = results['future_egomotions'].unsqueeze(0)
        bev_transform = results.get('aug_transform', None)
        labels['img_is_valid'] = results.get('img_is_valid', None)

        if bev_transform is not None:
            bev_transform = bev_transform.float()

        segmentation_labels = self.warper.cumulative_warp_features_reverse(
            segmentation_labels.float().unsqueeze(2),
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).long().contiguous()
        labels['segmentation'] = segmentation_labels
        future_distribution_inputs.append(segmentation_labels)

        # Warp instance labels to present's reference frame
        gt_instance = self.warper.cumulative_warp_features_reverse(
            gt_instance.float().unsqueeze(2),
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).long().contiguous()[:, :, 0]
        labels['instance'] = gt_instance

        instance_center_labels = self.warper.cumulative_warp_features_reverse(
            instance_center_labels,
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).contiguous()
        labels['centerness'] = instance_center_labels

        instance_offset_labels = self.warper.cumulative_warp_features_reverse(
            instance_offset_labels,
            future_egomotion[(self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).contiguous()
        labels['offset'] = instance_offset_labels

        future_distribution_inputs.append(instance_center_labels)
        future_distribution_inputs.append(instance_offset_labels)

        instance_flow_labels = self.warper.cumulative_warp_features_reverse(
            instance_flow_labels,
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).contiguous()
        labels['flow'] = instance_flow_labels
        future_distribution_inputs.append(instance_flow_labels)

        if len(future_distribution_inputs) > 0:
            future_distribution_inputs = torch.cat(
                future_distribution_inputs, dim=2)

        # self.visualizer.visualize_motion(labels=labels)
        # pdb.set_trace()

        return labels, future_distribution_inputs


@PIPELINES.register_module()
class CentripetalBackwardFlow(object):
    def __init__(self, grid_conf, ignore_index=255, instance_windows=3, only_vehicle=True, filter_invisible=True,
                 receptive_field=1):
        self.grid_conf = grid_conf
        # torch.tensor
        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
        )
        # convert numpy
        self.bev_resolution = self.bev_resolution.numpy()
        self.bev_start_position = self.bev_start_position.numpy()
        self.bev_dimension = self.bev_dimension.numpy()
        self.spatial_extent = (grid_conf['xbound'][1], grid_conf['ybound'][1])
        self.ignore_index = ignore_index
        self.only_vehicle = only_vehicle
        self.filter_invisible = filter_invisible
        self.instance_windows = instance_windows
        nusc_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        vehicle_classes = ['car', 'bus', 'construction_vehicle',
                           'bicycle', 'motorcycle', 'truck', 'trailer']

        self.vehicle_cls_ids = np.array([nusc_classes.index(
            cls_name) for cls_name in vehicle_classes])

        self.warper = FeatureWarper(grid_conf=grid_conf)
        self.receptive_field = receptive_field
        self.visualizer = Visualizer(out_dir="./figures/test", coordinate_system="ego")

    @staticmethod
    def generate_flow(flow, instance_img, instance_id):
        """
        Generate ground truth for the flow of each instance based on instance segmentation.
        """
        seq_len, h, w = instance_img.shape
        x, y = torch.meshgrid(torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float))
        grid = torch.stack((x, y), dim=0)  # (2,h,w)

        # flow with shape (seq,2,h,w)

        # Set the first frame
        instance_mask = (instance_img[0] == instance_id)
        # here x, y should be changed because of the loading method of matrix
        flow[0, 1, instance_mask] = grid[0, instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
        flow[0, 0, instance_mask] = grid[1, instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]
        # beyond set the vector from each instance grid to the center of this instance. The center of each instance is set by mean
        for i, timestep in enumerate(range(seq_len)):
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

        for i, instance_token in enumerate(self.total_visible_instance_set):  # iterate through each instance
            flow = self.generate_flow(flow, instance_img, instance_map[instance_token])
        return flow

    def __call__(self, results):
        # annotation_token ==> instance_id
        instance_map = {}
        self.total_visible_instance_set = set()

        # convert LiDAR bounding boxes to motion labels
        num_frame = len(results['gt_bboxes_3d'])
        all_gt_bboxes_3d = results['gt_bboxes_3d']
        all_gt_labels_3d = results['gt_labels_3d']
        all_instance_tokens = results['instance_tokens']
        all_vis_tokens = results['gt_vis_tokens']
        # 4x4 transformation matrix (if exist)
        bev_transform = results.get('aug_transform', None)

        segmentations = []
        instances = []

        # 对于 invalid frame: 所有 label 均为 255 -> 白色
        # 对于 valid frame: seg & instance 背景是 0(黑色），其它背景为255（白色）

        # 首先，创建一个instance_map，用于记录每个instance的id，并且我们只前N帧都存在的instance
        for window_i, frame_index in enumerate(range(num_frame)):
            gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d[frame_index], all_gt_labels_3d[frame_index]
            instance_tokens = all_instance_tokens[frame_index]
            vis_tokens = all_vis_tokens[frame_index]

            if gt_bboxes_3d is None:
                raise ValueError(
                    "invalid frame found in this sample, which is supposed to be filtered out in prepare_train_data")
            else:
                if self.only_vehicle:
                    vehicle_mask = np.isin(gt_labels_3d, self.vehicle_cls_ids)
                    gt_bboxes_3d = gt_bboxes_3d[vehicle_mask]
                    gt_labels_3d = gt_labels_3d[vehicle_mask]
                    instance_tokens = instance_tokens[vehicle_mask]
                    vis_tokens = vis_tokens[vehicle_mask]

                if self.filter_invisible:
                    visible_mask = (vis_tokens != 1)
                    gt_bboxes_3d = gt_bboxes_3d[visible_mask]
                    gt_labels_3d = gt_labels_3d[visible_mask]
                    instance_tokens = instance_tokens[visible_mask]

                all_gt_bboxes_3d[frame_index] = gt_bboxes_3d
                all_gt_labels_3d[frame_index] = gt_labels_3d
                all_instance_tokens[frame_index] = instance_tokens

                # valid sample and has objects
                if len(gt_bboxes_3d.tensor) > 0:
                    # here bbox_corners contains all bboxes in this frame
                    for index, instance_token in enumerate(instance_tokens):
                        if window_i >= self.instance_windows and instance_token not in self.total_visible_instance_set:
                            # if this instance is not visible in the previous windows, then we don't draw it
                            continue
                        self.total_visible_instance_set.add(instance_token)
                        if instance_token not in instance_map:
                            instance_map[instance_token] = len(
                                instance_map) + 1
        # 使用self.total_visible_instance_set进行二次过滤
        for window_i, frame_index in enumerate(range(num_frame)):
            gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d[frame_index], all_gt_labels_3d[frame_index]
            instance_tokens = all_instance_tokens[frame_index]
            token_mask = np.isin(instance_tokens, list(self.total_visible_instance_set))
            all_gt_bboxes_3d[frame_index] = gt_bboxes_3d[token_mask]
            all_gt_labels_3d[frame_index] = gt_labels_3d[token_mask]
            all_instance_tokens[frame_index] = instance_tokens[token_mask]
        # 接着，有一些instance有可能在连续的过程中突然消失，我们需要在那一帧的数据里给他重新添加上去
        # 我们用一个set来维护我们见到的instance，如果在某一帧中，我们没有见到这个instance，
        # 那么我们就把它之前帧的数据给他添加上去，这样就不会消失某一个instance在map中闪烁的情况

        #TODO: 这里的坐标变换有问题，需要重写

        # self.cur_visible_instance_set = set()
        # for window_i, frame_index in enumerate(range(num_frame)):
        #     gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d[frame_index], all_gt_labels_3d[frame_index]
        #     instance_tokens = all_instance_tokens[frame_index]
        #     if gt_bboxes_3d is None:
        #         raise ValueError(
        #             "invalid frame found in this sample, which is supposed to be filtered out in prepare_train_data")
        #     # 遍历self.cur_visible_instance_set，如果在当前帧中没有这个instance，那么我们就把它添加上去
        #     for should_have_instance_token in self.cur_visible_instance_set:
        #         if should_have_instance_token not in instance_tokens:
        #             # 找到前一帧数据
        #             prev_frame_index = frame_index - 1
        #             prev_gt_bboxe_3d = all_gt_bboxes_3d[prev_frame_index]
        #             prev_gt_labels_3d = all_gt_labels_3d[prev_frame_index]
        #             prev_instance_tokens = all_instance_tokens[prev_frame_index]
        #             # 找到这个token对应的位置
        #             token_index = np.where(prev_instance_tokens == should_have_instance_token)[0][0]
        #             temp_gt_bboxe_3d = prev_gt_bboxe_3d[token_index]
        #             temp_gt_labels_3d = prev_gt_labels_3d[token_index]
        #             gt_bboxes_3d = np.concatenate((gt_bboxes_3d, temp_gt_bboxe_3d[np.newaxis, :]), axis=0)
        #             gt_labels_3d = np.append(gt_labels_3d, temp_gt_labels_3d)
        #             instance_tokens = np.append(instance_tokens, should_have_instance_token)
        #
        #     all_gt_bboxes_3d[frame_index] = gt_bboxes_3d
        #     all_gt_labels_3d[frame_index] = gt_labels_3d
        #     all_instance_tokens[frame_index] = instance_tokens
        #
        #     for instance_token in instance_tokens:
        #         # 将当前帧的instance加入到当前帧的set中
        #         self.cur_visible_instance_set.add(instance_token)
        # 现在，开始创建segmentation map, instance map与 centri_backward_flow
        for window_i, frame_index in enumerate(range(num_frame)):
            gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d[frame_index], all_gt_labels_3d[frame_index]
            instance_tokens = all_instance_tokens[frame_index]

            if gt_bboxes_3d is None:
                raise ValueError(
                    "invalid frame found in this sample, which is supposed to be filtered out in prepare_train_data")
            else:
                # for valid samples
                segmentation = np.zeros(
                    (self.bev_dimension[1], self.bev_dimension[0]))
                instance = np.zeros(
                    (self.bev_dimension[1], self.bev_dimension[0]))

                # valid sample and has objects
                if len(gt_bboxes_3d.tensor) > 0:
                    bbox_corners = gt_bboxes_3d.corners[:, [0, 3, 7, 4], :2].numpy()
                    bbox_corners = np.round(
                        (bbox_corners - self.bev_start_position[:2] + self.bev_resolution[
                                                                      :2] / 2.0) / self.bev_resolution[:2]).astype(
                        np.int32)
                    # here bbox_corners contains all bboxes in this frame
                    for index, instance_token in enumerate(instance_tokens):
                        # instance_id start from 1
                        instance_id = instance_map[instance_token]
                        poly_region = bbox_corners[index]
                        cv2.fillPoly(segmentation, [poly_region], 1.0)
                        cv2.fillPoly(instance, [poly_region], instance_id)

            segmentations.append(segmentation)
            instances.append(instance)

        # segmentation = 1 where objects are located
        segmentations = torch.from_numpy(
            np.stack(segmentations, axis=0)).long()
        instances = torch.from_numpy(np.stack(instances, axis=0)).long()
        centri_backward_flow = self.get_flow_label(instances, instance_map, ignore_index=255)
        # generate heatmap & offset from segmentation & instance
        # shape of future_egomotion (num_frames,6)
        future_egomotions = results['future_egomotions'][- num_frame:]
        instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label_with_warper(
            instance_img=instances,
            future_egomotion=future_egomotions,
            num_instances=len(instance_map),
            ignore_index=self.ignore_index,
            subtract_egomotion=True,
            warper=self.warper,
            bev_transform=bev_transform,
        )

        invalid_mask = (segmentations[:, 0, 0] == self.ignore_index)
        instance_centerness[invalid_mask] = self.ignore_index

        # only keep detection labels for the current frame
        results['gt_bboxes_3d'] = all_gt_bboxes_3d[0]
        results['gt_labels_3d'] = all_gt_labels_3d[0]
        results['instance_tokens'] = all_instance_tokens[0]
        results['gt_valid_flag'] = results['gt_valid_flag'][0]
        results.update({
            'motion_segmentation': segmentations,  # (num_frames, bev_h,bev_w)
            'motion_instance': instances,  # (num_frames, bev_h,bev_w)  -> ground truth
            'instance_centerness': instance_centerness,  # (num_frames, 1, bev_h, bev_w)
            'instance_offset': instance_offset,  # (num_frames, 2, bev_h, bev_w)
            'instance_flow': instance_flow,  # (num_frames, 2, bev_h, bev_w)
        })
        self.visualization(results,save_path='./figures/test')
        assert False
        return results

    def visualization(self, results, save_path=None):
        motion_labels, _ = self.prepare_future_labels(results)
        self.visualizer.visualize_motion_gif(labels=motion_labels)
        #self.visualizer.visualize_gt_motion(motion_labels=motion_labels, save_path=save_path)

    def prepare_future_labels(self, results):
        labels = {}
        future_distribution_inputs = []

        segmentation_labels = results['motion_segmentation'].unsqueeze(0)
        instance_center_labels = results['instance_centerness'].unsqueeze(0)
        instance_offset_labels = results['instance_offset'].unsqueeze(0)
        instance_flow_labels = results['instance_flow'].unsqueeze(0)
        gt_instance = results['motion_instance'].unsqueeze(0)
        future_egomotion = results['future_egomotions'].unsqueeze(0)
        bev_transform = results.get('aug_transform', None)
        labels['img_is_valid'] = results.get('img_is_valid', None)

        if bev_transform is not None:
            bev_transform = bev_transform.float()

        segmentation_labels = self.warper.cumulative_warp_features_reverse(
            segmentation_labels.float().unsqueeze(2),
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).long().contiguous()
        labels['segmentation'] = segmentation_labels
        future_distribution_inputs.append(segmentation_labels)

        # Warp instance labels to present's reference frame
        gt_instance = self.warper.cumulative_warp_features_reverse(
            gt_instance.float().unsqueeze(2),
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).long().contiguous()[:, :, 0]
        labels['instance'] = gt_instance

        instance_center_labels = self.warper.cumulative_warp_features_reverse(
            instance_center_labels,
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).contiguous()
        labels['centerness'] = instance_center_labels

        instance_offset_labels = self.warper.cumulative_warp_features_reverse(
            instance_offset_labels,
            future_egomotion[(self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).contiguous()
        labels['offset'] = instance_offset_labels

        future_distribution_inputs.append(instance_center_labels)
        future_distribution_inputs.append(instance_offset_labels)

        instance_flow_labels = self.warper.cumulative_warp_features_reverse(
            instance_flow_labels,
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).contiguous()
        labels['flow'] = instance_flow_labels
        future_distribution_inputs.append(instance_flow_labels)

        if len(future_distribution_inputs) > 0:
            future_distribution_inputs = torch.cat(
                future_distribution_inputs, dim=2)

        # self.visualizer.visualize_motion(labels=labels)
        # pdb.set_trace()

        return labels, future_distribution_inputs
