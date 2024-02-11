import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import warnings

from .data_io import get_transform, read_all_lines
from .wrappers import Camera, Pose


def ref_points_generator(start, shape, voxel_size, normalize=True):
    min_x, min_y, min_z = start
    x_range = torch.arange(shape[0], dtype=torch.float) * voxel_size + voxel_size / 2 + min_x
    y_range = torch.arange(shape[1], dtype=torch.float) * voxel_size + voxel_size / 2 + min_y
    z_range = torch.arange(shape[2], dtype=torch.float) * voxel_size + voxel_size / 2 + min_z

    W, H, D = x_range.shape[0], y_range.shape[0], z_range.shape[0]
    # import pdb; pdb.set_trace()

    grid_x = x_range.view(-1, 1, 1).repeat(1, H, D)
    grid_y = y_range.view(1, -1, 1).repeat(W, 1, D)
    grid_z = z_range.view(1, 1, -1).repeat(W, H, 1)

    # grid_x = grid_x.view(1, 1, W, H, D)
    # grid_y = grid_y.view(1, 1, W, H, D)
    # grid_z = grid_z.view(1, 1, W, H, D)

    coords = torch.stack((grid_x, grid_y, grid_z), 3).float()  # [B,Coords=3,W,H,D]

    # # D, H, W
    # meshgrid = torch.meshgrid(min_z + z_range, min_y + y_range, min_x + x_range, indexing='ij')
    # z_coords, y_coords, x_coords = meshgrid
    # coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)

    if normalize:
        coords[..., 0] = (coords[..., 0] - torch.min(coords[..., 0])) / (
                torch.max(coords[..., 0]) - torch.min(coords[..., 0]) + 1e-30)
        coords[..., 1] = (coords[..., 1] - torch.min(coords[..., 1])) / (
                torch.max(coords[..., 1]) - torch.min(coords[..., 1]) + 1e-30)
        coords[..., 2] = (coords[..., 2] - torch.min(coords[..., 2])) / (
                torch.max(coords[..., 2]) - torch.min(coords[..., 2]) + 1e-30)

    return coords


class DSDataset(Dataset):
    def __init__(self, datapath, list_filename, training, transform=True):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(
            list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None
        self.transform = transform

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
        left_img = self.load_image(os.path.join(
            self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(
            self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(
            self.datapath, self.disp_filenames[index]))

        w, h = left_img.size
        crop_w, crop_h = 880, 400

        processed = get_transform()

        if self.transform:
            if w < crop_w:
                left_img = processed(left_img).numpy()
                right_img = processed(right_img).numpy()

                left_img = np.lib.pad(
                    left_img, ((0, 0), (0, 0), (0, crop_w-w)), mode='constant', constant_values=0)
                right_img = np.lib.pad(
                    right_img, ((0, 0), (0, 0), (0, crop_w-w)), mode='constant', constant_values=0)
                disparity = np.lib.pad(
                    disparity, ((0, 0), (0, crop_w-w)), mode='constant', constant_values=0)

                left_img = torch.Tensor(left_img)
                right_img = torch.Tensor(right_img)
            else:
                left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
                right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
                disparity = disparity[h - crop_h:h, w - crop_w: w]

                left_img = processed(left_img)
                right_img = processed(right_img)
        else:
            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img)

        return {"left": left_img,
                "right": right_img,
                "disparity": disparity,
                "left_filename": self.left_filenames[index]}


class VoxelDSDatasetCalib(Dataset):
    def __init__(self, datapath, list_filename, training, roi_scale, voxel_sizes, transform=True, *,
                 filter_ground=True):
        self.datapath = datapath
        # pre-computed gt label
        self.stored_gt = False
        self.left_filenames, self.right_filenames, self.depth_filenames, self.gt_filenames, self.calib_filenames = \
            self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.depth_filenames is not None

        # Camera intrinsics
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
        # if ground y > 1 will be filtered
        self.filter_ground = filter_ground

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = []
        right_images = []
        calib = []
        for x in splits:
            left_images.append(x[0])
            right_images.append(x[1])
            calib.append(x[-1])
        if len(splits[0]) == 3:  # ground truth not available
            return left_images, right_images, None, None, calib
        elif len(splits[0]) == 4:
            depth_map = [x[2] for x in splits]
            return left_images, right_images, depth_map, None, calib
        elif len(splits[0]) == 5:
            self.stored_gt = True
            depth_map = []
            gt_label = []
            for x in splits:
                depth_map.append(x[2])
                gt_label.append(x[3])
            return left_images, right_images, depth_map, gt_label, calib
        else:
            raise RuntimeError('Dataset filename format not supported.')

    @staticmethod
    def load_image(filename):
        return Image.open(filename).convert('RGB')

    @staticmethod
    def load_depth(filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    @staticmethod
    def load_gt(filename):
        return torch.load(filename)

    def load_calib(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        R_101 = None
        T_101 = None
        P_rect_101 = None
        R_rect_101 = None
        R_103 = None
        T_103 = None
        P_rect_103 = None
        R_rect_103 = None
        for line in lines:
            splits = line.split()
            if splits[0] == 'R_101:':
                R_101 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'T_101:':
                T_101 = np.array(list(map(float, splits[1:])))
            elif splits[0] == 'P_rect_101:':
                P_rect_101 = np.array(list(map(float, splits[1:]))).reshape(3, 4)
            elif splits[0] == 'R_rect_101:':
                R_rect_101 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'R_103:':
                R_103 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'T_103:':
                T_103 = np.array(list(map(float, splits[1:])))
            elif splits[0] == 'P_rect_103:':
                P_rect_103 = np.array(list(map(float, splits[1:]))).reshape(3, 4)
            elif splits[0] == 'R_rect_103:':
                R_rect_103 = np.array(list(map(float, splits[1:]))).reshape(3, 3)

        # 4x4
        Rt_101 = np.concatenate([R_101, np.expand_dims(T_101, axis=-1)], axis=-1)
        Rt_101 = np.concatenate([Rt_101, np.array([[0., 0., 0., 1.]])], axis=0)
        Rt_103 = np.concatenate([R_103, np.expand_dims(T_103, axis=-1)], axis=-1)
        Rt_103 = np.concatenate([Rt_103, np.array([[0., 0., 0., 1.]])], axis=0)

        R_rect_101 = np.concatenate([R_rect_101, np.array([[0., 0., 0.]]).T], axis=-1)
        R_rect_101 = np.concatenate([R_rect_101, np.array([[0., 0., 0., 1.]])], axis=0)
        R_rect_103 = np.concatenate([R_rect_103, np.array([[0., 0., 0.]]).T], axis=-1)
        R_rect_103 = np.concatenate([R_rect_103, np.array([[0., 0., 0., 1.]])], axis=0)

        # T_world_cam_101 = P_rect_101 @ R_rect_101 @ Rt_101
        T_world_cam_101 = R_rect_101 @ Rt_101
        T_world_cam_101 = np.concatenate([T_world_cam_101[:3, :3].flatten(), T_world_cam_101[:3, 3]], axis=-1)
        # T_world_cam_103 = P_rect_103 @ R_rect_103 @ Rt_103
        T_world_cam_103 = R_rect_103 @ Rt_103
        T_world_cam_103 = np.concatenate([T_world_cam_103[:3, :3].flatten(), T_world_cam_103[:3, 3]], axis=-1)

        self.c_u = P_rect_101[0, 2]
        self.c_v = P_rect_101[1, 2]
        self.f_u = P_rect_101[0, 0]
        self.f_v = P_rect_101[1, 1]

        cam_101 = np.array([P_rect_101[0, 0], P_rect_101[1, 1], P_rect_101[0, 2], P_rect_101[1, 2]])
        cam_103 = np.array([P_rect_103[0, 0], P_rect_103[1, 1], P_rect_103[0, 2], P_rect_103[1, 2]])

        T_world_cam_101 = T_world_cam_101.astype(np.float32)
        cam_101 = cam_101.astype(np.float32)
        T_world_cam_103 = T_world_cam_103.astype(np.float32)
        cam_103 = cam_103.astype(np.float32)

        self.lidar_extrinsic = Pose(T_world_cam_101)

        return T_world_cam_101, cam_101, T_world_cam_103, cam_103

    def ref_point_mask(self, img_size, cam_intrinsic, extrinsic, level):
        T_world_cam = Pose(extrinsic)
        cam = Camera(cam_intrinsic)
        grid_size = self.grid_sizes[level]
        # shape may be changed later
        # ref_interval = ref_interval_generator([-16, -31, 0], [grid_size, grid_size, grid_size], voxel_size).view(-1, 3)
        ref_points = ref_points_generator([self.roi_scale[0], self.roi_scale[2], self.roi_scale[4]], grid_size,
                                          self.voxel_sizes[level], normalize=False).view(-1, 3)
        # interval_coord, _ = cam.project(T_world_cam.transform(ref_interval))
        ref_coord, _ = cam.project(T_world_cam.transform(ref_points))

        # interval_x = (interval_coord[:, 0] >= 0) & (interval_coord[:, 0] <= img_size[0])
        # interval_y = (interval_coord[:, 1] >= 0) & (interval_coord[:, 1] <= img_size[1])
        ref_x = (ref_coord[:, 0] >= 0) & (ref_coord[:, 0] <= img_size[0])
        ref_y = (ref_coord[:, 1] >= 0) & (ref_coord[:, 1] <= img_size[1])
        ref_mask = ref_x & ref_y
        '''
        interval_mask = (
                torch.nonzero(
                    (interval_x & interval_y).view(grid_size - 1, grid_size - 1, grid_size - 1)) + .5).unsqueeze(
            -2).repeat(1, 8, 1)
        oct_index = torch.tensor([-.5, .5])
        oct_indices = torch.cartesian_prod(oct_index, oct_index, oct_index)

        valid_indices = (interval_mask + oct_indices).to(int).view(-1, 3)
        ref_volume = torch.zeros([grid_size, grid_size, grid_size])
        ref_volume[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1
        ref_volume = ref_volume.to(bool).view(-1)
        '''
        # return ref_volume | ref_mask
        return ref_mask

    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        x = (uv_depth[:, 0] - self.c_u) * uv_depth[:, 2] / self.f_u
        y = (uv_depth[:, 1] - self.c_v) * uv_depth[:, 2] / self.f_v
        pts_3d_rect = np.zeros_like(uv_depth)
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2:] = uv_depth[:, 2:]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        return self.lidar_extrinsic.inverse().transform(self.project_image_to_rect(uv_depth)).numpy()

    def calc_cloud(self, depth, left_img=None):
        mask = (depth > 0).reshape(-1)
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth], axis=-1)
        points = points.reshape(-1, 3)
        points = points[mask]
        cloud = self.project_image_to_velo(points)
        if left_img is not None:
            left_img = left_img.reshape(-1, 3)
            return np.concatenate([cloud, left_img[mask]], axis=-1)

        return cloud

    def filter_cloud(self, cloud):
        min_mask = cloud >= [self.roi_scale[0], self.roi_scale[2], self.roi_scale[4]]
        if self.filter_ground and self.roi_scale[3] > 1:
            max_mask = cloud <= [self.roi_scale[1], 1, self.roi_scale[5]]
        else:
            max_mask = cloud <= [self.roi_scale[1], self.roi_scale[3], self.roi_scale[5]]
        min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
        max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
        filter_mask = min_mask & max_mask
        filtered_cloud = cloud[filter_mask]
        return filtered_cloud

    def calc_voxel_grid(self, filtered_cloud, level, parent_grid=None, occupied_gate=20):
        assert occupied_gate > 0

        vox_size = self.voxel_sizes[level]
        reference_points = ref_points_generator([self.roi_scale[0], self.roi_scale[2], self.roi_scale[4]],
                                                self.grid_sizes[level], vox_size, normalize=False).view(-1, 3).numpy()

        if parent_grid is not None:
            search_mask = parent_grid[:, None, :, None, :, None].repeat(1, 2, 1, 2, 1, 2).view(-1).to(
                bool).numpy()
        else:
            search_mask = torch.ones(reference_points.shape[0]).to(bool)

        vox_hits = np.bitwise_and.reduce(
            np.abs(filtered_cloud[..., None, :] - reference_points[search_mask]) <= vox_size / 2,
            axis=-1)
        vox_hits = np.sum(vox_hits, axis=0) >= occupied_gate
        occupied_grid = np.zeros(reference_points.shape[0])
        occupied_grid[search_mask] = vox_hits.astype(int)

        return occupied_grid.reshape(*self.grid_sizes[level]), reference_points[occupied_grid.astype(bool)]

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img_ = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img_ = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        T_world_cam_101, cam_101, T_world_cam_103, cam_103 = self.load_calib(
            os.path.join(self.datapath, self.calib_filenames[index]))
        depth_gt = self.load_depth(os.path.join(self.datapath, self.depth_filenames[index]))

        # numpy to tensor
        T_world_cam_101 = torch.from_numpy(T_world_cam_101)
        T_world_cam_103 = torch.from_numpy(T_world_cam_103)

        w, h = left_img_.size
        crop_w, crop_h = 880, 400

        processed = get_transform()
        left_top = [0, 0]

        if self.transform:
            if w < crop_w:
                left_img = processed(left_img_).numpy()
                right_img = processed(right_img_).numpy()

                w_pad = crop_w - w
                left_img = np.lib.pad(
                    left_img, ((0, 0), (0, 0), (0, w_pad)), mode='constant', constant_values=0)
                right_img = np.lib.pad(
                    right_img, ((0, 0), (0, 0), (0, w_pad)), mode='constant', constant_values=0)
                depth_gt = np.lib.pad(
                    depth_gt, ((0, 0), (0, w_pad)), mode='constant', constant_values=0)

                left_img = torch.Tensor(left_img)
                right_img = torch.Tensor(right_img)
            else:
                w_crop = w - crop_w
                h_crop = h - crop_h
                left_img_ = left_img_.crop((w_crop, h_crop, w, h))
                right_img_ = right_img_.crop((w_crop, h_crop, w, h))
                depth_gt = depth_gt[h_crop: h, w_crop: w]

                left_img = processed(left_img_)
                right_img = processed(right_img_)
                left_top = [w_crop, h_crop]
        else:
            w_crop = w - crop_w
            h_crop = h - crop_h
            left_img_ = left_img_.crop((w_crop, h_crop, w, h))
            right_img_ = right_img_.crop((w_crop, h_crop, w, h))
            left_img = np.asarray(left_img_)
            right_img = np.asarray(right_img_)
            left_top = [w_crop, h_crop]

        left_top = np.repeat(np.array([left_top]), repeats=2, axis=0)

        # canvas = np.zeros((400, 880, 3), dtype=np.float32)
        # left_img_ = np.asarray(left_img_)
        # canvas[:left_img_.shape[0], :left_img_.shape[1], :] = left_img_
        colored_cloud_gt = self.calc_cloud(depth_gt)  # , left_img=canvas)
        filtered_cloud_gt = self.filter_cloud(colored_cloud_gt[..., :3])

        if self.stored_gt:
            all_vox_grid_gt = self.load_gt(os.path.join(self.datapath, self.gt_filenames[index]))
            valid_gt, _ = self.calc_voxel_grid(filtered_cloud_gt, 0)
            if not torch.allclose(all_vox_grid_gt[0], torch.from_numpy(valid_gt)):
                warnings.warn(
                    f'Stored label inconsistent.\n Loaded gt: \n {all_vox_grid_gt[0]} \n Validate gt: \n {valid_gt}')
        else:
            all_vox_grid_gt = []
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

        ref_masks = []
        '''
        for _ in range(len(self.grid_sizes)):
            mask_101 = self.ref_point_mask([imw, imh], cam_101, T_world_cam_101, _)
            # right cam projection may not be needed
            # mask_103 = self.ref_point_mask([imw, imh], cam_103, T_world_cam_103, _)
            ref_masks.append(mask_101)  # | mask_103)
        '''

        return {"left": left_img,
                "right": right_img,
                'T_world_cam_101': T_world_cam_101,
                'cam_101': cam_101,
                'T_world_cam_103': T_world_cam_103,
                'cam_103': cam_103,
                "depth": depth_gt,
                "voxel_grid": all_vox_grid_gt,
                'point_cloud': colored_cloud_gt.astype(np.float32).tobytes(),
                'filtered_point_cloud': filtered_cloud_gt.astype(np.float32).tobytes(),
                'ref_masks': ref_masks,
                'left_top': left_top,
                "left_filename": self.left_filenames[index]}
