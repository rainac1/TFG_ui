import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
import os
import numpy as np
import json


class TorsoDataset(Dataset):
    def __init__(self, path, transform_inp, transform_out):
        self.path = path
        self.transform_inp = transform_inp
        self.transform_out = transform_out
        img_list = sorted(os.listdir(os.path.join(path, 'frame_imgs')))
        self.tgt_list = [os.path.join(path, 'frame_imgs', imgpath) for imgpath in img_list]
        inp_list = sorted(os.listdir(os.path.join(path, 'logs', 'stage2', 'results')))
        self.inp_list = [os.path.join(path, 'logs', 'stage2', 'results', imgpath) for imgpath in inp_list if 'rgb' in imgpath]
        transform_paths = [os.path.join(path, f'transforms_train.json'),
                           os.path.join(path, f'transforms_test.json'),
                           os.path.join(path, f'transforms_val.json')]
        frames = None
        for transform_path in transform_paths:
            with open(transform_path, 'r') as f:
                tmp_transform = json.load(f)
                if frames is None:
                    frames = tmp_transform['frames']
                else:
                    frames.extend(tmp_transform['frames'])

        poses = np.array([f['transform_matrix'] for f in frames], dtype=np.float32).reshape(-1, 4, 4)
        self.poses = torch.from_numpy(self.mat2angle(poses)).float()
        self.length = min(len(self.tgt_list), len(self.inp_list))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inp = Image.open(self.inp_list[index])
        tgt = Image.open(self.tgt_list[index])
        inp = self.transform_inp(inp)
        tgt = self.transform_out(tgt)
        pose = self.poses[index]
        return inp, tgt, pose

    def mat2angle(self, mat):
        trans = mat[:, :3, -1]
        mat = mat[:, :3, :3]
        sy = np.sqrt(mat[..., 0, 0] ** 2 + mat[..., 1, 0] ** 2)
        singular = sy < 1e-6
        x1 = mat[..., 2, 1] * ~singular + -mat[..., 1, 2] * singular
        x2 = mat[..., 2, 2] * ~singular + -mat[..., 1, 1] * singular
        x = np.arctan2(x1, x2)
        y = np.arctan2(-mat[..., 2, 0], sy)
        z = np.arctan2(mat[..., 1, 0], mat[..., 0, 0])
        z = z * ~singular
        angle = np.stack([x, y, z], -1)
        return np.concatenate([angle, trans], -1)
