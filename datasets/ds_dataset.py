import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os

from .data_io import get_transform, read_all_lines


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
