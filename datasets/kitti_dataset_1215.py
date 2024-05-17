import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

from .data_io import get_transform, read_all_lines
from .wrappers import Camera, Pose
from .ds_dataset import ref_points_generator


class VoxelDataset(Dataset):
    def __init__(self, datapath, roi_scale, voxel_sizes, transform, *, filter_ground, color_jitter, occupied_gates):
        self.datapath = datapath
        self.stored_gt = False
        # initialize as null
        self.c_u = None
        self.c_v = None
        self.f_u = None
        self.f_v = None
        self.lidar_extrinsic = None
        self.roi_scale = roi_scale  # [min_x, max_x, min_y, max_y, min_z, max_z]
        assert len(voxel_sizes) == 4, 'Incomplete voxel sizes for 4 levels.'
        self.voxel_sizes = voxel_sizes

        self.grid_sizes = []
        for voxel_size in self.voxel_sizes:
            range_x = self.roi_scale[1] - self.roi_scale[0]
            range_y = self.roi_scale[3] - self.roi_scale[2]
            range_z = self.roi_scale[5] - self.roi_scale[4]
            if range_x % voxel_size != 0 or range_y % voxel_size != 0 or range_z % voxel_size != 0:
                raise RuntimeError('Voxel volume range indivisible by voxel sizes.')

            grid_size_x = int(range_x // voxel_size)
            grid_size_y = int(range_y // voxel_size)
            grid_size_z = int(range_z // voxel_size)
            self.grid_sizes.append((grid_size_x, grid_size_y, grid_size_z))

        self.transform = transform
        # if ground y > ground_y will be filtered
        self.filter_ground = filter_ground
        self.ground_y = None
        self.color_jitter = color_jitter
        self.occupied_gates = occupied_gates

    def load_path(self, list_filename):
        raise NotImplementedError

    @staticmethod
    def load_image(filename):
        return Image.open(filename).convert('RGB')

    @staticmethod
    def load_disp(filename):
        # 16 bit Grayscale
        data = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        out = data.astype(np.float32) / 256.
        return out

    load_depth = load_disp

    @staticmethod
    def load_flow(filename):
        raise NotImplementedError

    @staticmethod
    def load_gt(filename):
        return torch.load(filename)

    def load_calib(self, filename):
        raise NotImplementedError

    def project_image_to_rect(self, uv_depth):
        x = (uv_depth[:, 0] - self.c_u) * uv_depth[:, 2] / self.f_u
        y = (uv_depth[:, 1] - self.c_v) * uv_depth[:, 2] / self.f_v
        pts_3d_rect = np.zeros_like(uv_depth)
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        return self.lidar_extrinsic.inverse().transform(self.project_image_to_rect(uv_depth)).numpy()

    def filter_cloud(self, cloud):
        min_mask = cloud[..., :3] >= [self.roi_scale[0], self.roi_scale[2], self.roi_scale[4]]
        if self.filter_ground and self.roi_scale[3] > self.ground_y:
            max_mask = cloud[..., :3] <= [self.roi_scale[1], self.ground_y, self.roi_scale[5]]
        else:
            max_mask = cloud[..., :3] <= [self.roi_scale[1], self.roi_scale[3], self.roi_scale[5]]
        min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
        max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
        filter_mask = min_mask & max_mask
        filtered_cloud = cloud[filter_mask]
        return filtered_cloud

    def calc_voxel_grid(self, filtered_cloud, level, parent_grid=None, get_flow=False,
                        *,
                        rtol: float = 0.3):
        occupied_gate_ = self.occupied_gates[level]
        occupied_gate = occupied_gate_ if occupied_gate_ is not None else 1
        assert occupied_gate > 0

        vox_size = self.voxel_sizes[level]
        reference_points = ref_points_generator([self.roi_scale[0], self.roi_scale[2], self.roi_scale[4]],
                                                self.grid_sizes[level], vox_size, normalize=False).view(-1, 3).numpy()

        if parent_grid is not None:
            search_mask = parent_grid[:, None, :, None, :, None].repeat(1, 2, 1, 2, 1, 2).view(-1).to(
                bool).numpy()
        else:
            search_mask = torch.ones(reference_points.shape[0]).to(bool)

        # num_search_grids, num_pc - bool
        vox_hits = np.bitwise_and.reduce(
            np.abs(filtered_cloud[..., None, :3] - reference_points[search_mask]) <= vox_size / 2,
            axis=-1)
        # num_search_grids - bool
        valid_hits = np.sum(vox_hits, axis=0) >= occupied_gate
        occupied_grid = np.zeros(reference_points.shape[0])
        occupied_grid[search_mask] = valid_hits.astype(int)

        if not get_flow:
            return occupied_grid.reshape(*self.grid_sizes[level]), reference_points[occupied_grid.astype(bool)]
        else:
            assert filtered_cloud.shape[-1] == 6
            mean_flow = vox_hits.T @ filtered_cloud[..., 3:] / (np.sum(vox_hits, axis=0, keepdims=True).T + 1e-5)
            mean_flow = np.round(mean_flow, decimals=1)
            sflow = np.zeros(reference_points.shape)
            sflow[search_mask] = (mean_flow - rtol * np.sign(mean_flow) * vox_size) // vox_size * vox_size
            sflow *= occupied_grid[..., None]

            return occupied_grid.reshape(*self.grid_sizes[level]), reference_points[
                occupied_grid.astype(bool)], sflow.reshape(*self.grid_sizes[level], 3)


class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                  constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant',
                                       constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}


class VoxelKITTIDataset(VoxelDataset):
    def __init__(self, datapath, list_filename, training, roi_scale, voxel_sizes, transform=True, *,
                 filter_ground=True, color_jitter=False, occupied_gates=(20, 20, 10, 5)):
        super().__init__(datapath, roi_scale, voxel_sizes, transform, filter_ground=filter_ground,
                         color_jitter=color_jitter, occupied_gates=occupied_gates)
        self.left_filenames = None
        self.right_filenames = None
        self.disp_filenames = None
        self.gt_voxel_filenames = None
        self.calib_filenames = None
        self.load_path(list_filename)
        if training:
            assert self.disp_filenames is not None

        # Camera intrinsics
        self.baseline = 0.54
        self.ground_y = 1.5

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        self.left_filenames = []
        self.right_filenames = []
        self.calib_filenames = []
        for x in splits:
            self.left_filenames.append(x[0])
            self.right_filenames.append(x[1])
            self.calib_filenames.append(x[-1])

        # with gt disp and flow
        if len(splits[0]) >= 4:
            self.disp_filenames = [x[2] for x in splits]

            # stored gt available
            if len(splits[0]) > 4:
                self.stored_gt = True
                self.gt_voxel_filenames = [x[3] for x in splits]

    def load_calib(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        R_02 = None
        T_02 = None
        P_rect_02 = None
        R_rect_02 = None
        R_03 = None
        T_03 = None
        P_rect_03 = None
        R_rect_03 = None
        for line in lines:
            splits = line.split()
            if splits[0] == 'R_00:':
                R_02 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'T_00:':
                T_02 = np.array(list(map(float, splits[1:])))
            elif splits[0] == 'P_rect_00:':
                P_rect_02 = np.array(list(map(float, splits[1:]))).reshape(3, 4)
            elif splits[0] == 'R_rect_00:':
                R_rect_02 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'R_01:':
                R_03 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'T_01:':
                T_03 = np.array(list(map(float, splits[1:])))
            elif splits[0] == 'P_rect_03:':
                P_rect_03 = np.array(list(map(float, splits[1:]))).reshape(3, 4)
            elif splits[0] == 'R_rect_03:':
                R_rect_03 = np.array(list(map(float, splits[1:]))).reshape(3, 3)

        # 4x4
        Rt_02 = np.concatenate([R_02, np.expand_dims(T_02, axis=-1)], axis=-1)
        Rt_02 = np.concatenate([Rt_02, np.array([[0., 0., 0., 1.]])], axis=0)
        Rt_03 = np.concatenate([R_03, np.expand_dims(T_03, axis=-1)], axis=-1)
        Rt_03 = np.concatenate([Rt_03, np.array([[0., 0., 0., 1.]])], axis=0)

        R_rect_02 = np.concatenate([R_rect_02, np.array([[0., 0., 0.]]).T], axis=-1)
        R_rect_02 = np.concatenate([R_rect_02, np.array([[0., 0., 0., 1.]])], axis=0)
        R_rect_03 = np.concatenate([R_rect_03, np.array([[0., 0., 0.]]).T], axis=-1)
        R_rect_03 = np.concatenate([R_rect_03, np.array([[0., 0., 0., 1.]])], axis=0)

        T_world_cam_02 = R_rect_02 @ Rt_02
        T_world_cam_02 = np.concatenate([T_world_cam_02[:3, :3].flatten(), T_world_cam_02[:3, 3]], axis=-1)
        T_world_cam_03 = R_rect_03 @ Rt_03
        T_world_cam_03 = np.concatenate([T_world_cam_03[:3, :3].flatten(), T_world_cam_03[:3, 3]], axis=-1)

        self.c_u = P_rect_02[0, 2]
        self.c_v = P_rect_02[1, 2]
        self.f_u = P_rect_02[0, 0]
        self.f_v = P_rect_02[1, 1]

        cam_02 = np.array([P_rect_02[0, 0], P_rect_02[1, 1], P_rect_02[0, 2], P_rect_02[1, 2]])
        cam_03 = np.array([P_rect_03[0, 0], P_rect_03[1, 1], P_rect_03[0, 2], P_rect_03[1, 2]])

        T_world_cam_101 = T_world_cam_02.astype(np.float32)
        cam_101 = cam_02.astype(np.float32)
        T_world_cam_103 = T_world_cam_03.astype(np.float32)
        cam_103 = cam_03.astype(np.float32)

        self.lidar_extrinsic = Pose(T_world_cam_101)

        return T_world_cam_101, cam_101, T_world_cam_103, cam_103

    def calc_cloud(self, disparity):
        depth_gt = self.f_u * self.baseline / (disparity + 1e-5)
        mask = (disparity > 0).reshape(-1)

        rows, cols = depth_gt.shape
        x, y = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))

        points = np.stack([x, y, depth_gt], axis=-1).reshape(-1, 3)
        points = points[mask]

        cloud = self.project_image_to_velo(points)
        return cloud

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        T_world_cam_101, cam_101, T_world_cam_103, cam_103 = self.load_calib(
            os.path.join(self.datapath, self.calib_filenames[index]))
        disp_gt = None
        if self.disp_filenames is not None:
            disp_gt = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        # numpy to tensor
        T_world_cam_101 = torch.from_numpy(T_world_cam_101)
        T_world_cam_103 = torch.from_numpy(T_world_cam_103)

        w, h = left_img.size
        crop_w, crop_h = 1224, 370

        processed = get_transform()
        left_top = [0, 0]

        if self.transform:
            if w < crop_w:
                left_img = processed(left_img).numpy()
                right_img = processed(right_img).numpy()

                w_pad = crop_w - w
                left_img = np.lib.pad(
                    left_img, ((0, 0), (0, 0), (0, w_pad)), mode='constant', constant_values=0)
                right_img = np.lib.pad(
                    right_img, ((0, 0), (0, 0), (0, w_pad)), mode='constant', constant_values=0)
                if disp_gt is not None:
                    disp_gt = np.lib.pad(disp_gt, ((0, 0), (0, w_pad)), mode='constant', constant_values=0)

                left_img = torch.Tensor(left_img)
                right_img = torch.Tensor(right_img)
            else:
                w_crop = w - crop_w
                h_crop = h - crop_h
                left_img = left_img.crop((w_crop, h_crop, w, h))
                right_img = right_img.crop((w_crop, h_crop, w, h))
                if disp_gt is not None:
                    disp_gt = disp_gt[h_crop: h, w_crop: w]

                left_img = processed(left_img)
                right_img = processed(right_img)
                left_top = [w_crop, h_crop]
        else:
            w_crop = w - crop_w
            h_crop = h - crop_h
            left_img = left_img.crop((w_crop, h_crop, w, h))
            right_img = right_img.crop((w_crop, h_crop, w, h))
            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img)
            left_top = [w_crop, h_crop]

        left_top = np.repeat(np.array([left_top]), repeats=2, axis=0)

        filtered_cloud_gt = None
        all_vox_grid_gt = []
        if disp_gt is not None:
            cloud_gt = self.calc_cloud(disp_gt)
            filtered_cloud_gt = self.filter_cloud(cloud_gt)

            if self.stored_gt:
                all_vox_grid_gt = self.load_gt(os.path.join(self.datapath, self.gt_voxel_filenames[index]))
                # ===== Different occlusion handling technique when generating gt labels =====
                # valid_gt, _ = self.calc_voxel_grid(filtered_cloud_gt, 0)
                # if not torch.allclose(all_vox_grid_gt[0], torch.from_numpy(valid_gt)):
                #     warnings.warn(
                #         f'Stored label inconsistent.\n Loaded gt: \n {all_vox_grid_gt[0]} \n Validate gt: \n'
                #         f'{valid_gt}')
            else:
                parent_grid = None
                try:
                    for level in range(len(self.grid_sizes)):
                        vox_grid_gt, cloud_np_gt = self.calc_voxel_grid(
                            filtered_cloud_gt, level=level, parent_grid=parent_grid)
                        vox_grid_gt = torch.from_numpy(vox_grid_gt)

                        parent_grid = vox_grid_gt
                        all_vox_grid_gt.append(vox_grid_gt)
                except Exception as e:
                    raise RuntimeError('Error in calculating voxel grids from point cloud')

        imc, imh, imw = left_img.shape
        cam_101 = np.concatenate(([imw, imh], cam_101)).astype(np.float32)
        cam_103 = np.concatenate(([imw, imh], cam_103)).astype(np.float32)

        return {'left': left_img,
                'right': right_img,
                'T_world_cam_101': T_world_cam_101,
                'cam_101': cam_101,
                'T_world_cam_103': T_world_cam_103,
                'cam_103': cam_103,
                'voxel_grid': all_vox_grid_gt if len(all_vox_grid_gt) >= 0 else 'null',
                'point_cloud': filtered_cloud_gt.astype(
                    np.float32).tobytes() if filtered_cloud_gt is not None else 'null',
                'left_top': left_top,
                "left_filename": self.left_filenames[index]}
