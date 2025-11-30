import os
import cv2
import json
import re
import tqdm
import numpy as np

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from .utils import get_rays


def smooth_camera_path(poses, kernel_size=5):
    from scipy.spatial.transform import Rotation
    # smooth the camera trajectory...
    # poses: [N, 4, 4], numpy array
    N = poses.shape[0]
    K = kernel_size // 2

    trans = poses[:, :3, 3].copy()  # [N, 3]
    rots = poses[:, :3, :3].copy()  # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(0)
        poses[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()

    return poses


def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    pose1 = np.eye(4, dtype=pose.dtype)
    pose1[:3, :3] = pose[:3, :3].T
    pose1[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]  # pose1 = pose.inverse()
    pose1[:3, 3] = scale * pose1[:3, 3] + offset
    pose2 = np.eye(4, dtype=pose.dtype)
    pose2[:3, :3] = pose1[:3, :3].T
    pose2[:3, 3] = -pose1[:3, :3].T @ pose1[:3, 3]  # pose2 = pose.inverse()
    return pose2.astype(np.float32)


def polygon_area(x, y):
    x_ = x - x.mean()
    y_ = y - y.mean()
    correction = x_[-1] * y_[0] - y_[-1] * x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5 * np.abs(main_area + correction)


class NeRFDataset:
    def __init__(self, opt, device, mode='train', downscale=1):
        super().__init__()

        self.opt = opt
        self.device = device
        self.mode = mode  # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload  # 0 = disk, 1 = cpu, 2 = gpu
        self.bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16

        self.training = self.mode in ['train', 'all', 'trainval', 'val']
        self.num_rays = self.opt.num_rays if self.training else -1

        if mode == 'all':
            transform_paths = [os.path.join(self.root_path, f'transforms_train.json'),
                               os.path.join(self.root_path, f'transforms_test.json'),
                               os.path.join(self.root_path, f'transforms_val.json')]
            transform = None
            for transform_path in transform_paths:
                with open(transform_path, 'r') as f:
                    tmp_transform = json.load(f)
                    if transform is None:
                        transform = tmp_transform
                    else:
                        transform['frames'].extend(tmp_transform['frames'])
        # load train and val split
        elif mode == 'trainval':
            transform_paths = [os.path.join(self.root_path, f'transforms_train.json'),
                               os.path.join(self.root_path, f'transforms_val.json')]
            transform = None
            for transform_path in transform_paths:
                with open(transform_path, 'r') as f:
                    tmp_transform = json.load(f)
                    if transform is None:
                        transform = tmp_transform
                    else:
                        transform['frames'].extend(tmp_transform['frames'])
        # only load one specified split
        else:
            # no test, use val as test
            _split = 'val' if mode == '' else mode
            with open(os.path.join(self.root_path, f'transforms_{_split}.json'), 'r') as f:
                transform = json.load(f)
        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            self.H = int(transform['cy']) * 2 // downscale
            self.W = int(transform['cx']) * 2 // downscale
        # load cam intrinsics
        if 'focal_len' in transform:
            fl_x = fl_y = transform['focal_len']
        elif 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')
        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        K = torch.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fl_x, fl_y, cx, cy
        # load aabb
        bound = np.array(transform["bound"])
        aabb = 1.5 * (bound - bound.mean(0)) + bound.mean(0)  # [2, 3]
        self.scale = self.bound / (aabb[1] - aabb[0]).max()
        self.offset = -self.scale * aabb[1] + [.5 * self.bound, 0.5 * self.bound, 0.5 * self.bound]
        self.aabb = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)
        self.aabb = aabb * self.scale + self.offset[None]
        print(self.scale, self.offset, self.aabb)
        # load pre-extracted background image (should be the same size as training image...)
        if self.opt.bg_img == 'white':  # special
            bg_img = np.ones((self.H, self.W, 3), dtype=np.float32)
        elif self.opt.bg_img == 'black':  # special
            bg_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        else:  # load from file
            # default bg
            if self.opt.bg_img == '':
                self.opt.bg_img = os.path.join(self.root_path, 'bg.png')
            bg_img = cv2.imread(self.opt.bg_img, cv2.IMREAD_UNCHANGED)  # [H, W, 3]
            if bg_img.shape[0] != self.H or bg_img.shape[1] != self.W:
                bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = bg_img.astype(np.float32) / 255  # [H, W, 3/4]
        self.bg_img = bg_img
        # load data
        frames = transform["frames"]
        print(f'[INFO] load {len(frames)} {mode} frames.')
        if mode == 'train':
            if self.opt.part:
                frames = frames[::10]  # 1/10 frames
            elif self.opt.part2:
                frames = frames[:375]  # first 15s
        elif mode == 'val':
            frames = frames[:100]  # first 100 frames for val

        poses = np.array([f['transform_matrix'] for f in frames], dtype=np.float32).reshape(-1, 4, 4)
        self.poses = np.stack([nerf_matrix_to_ngp(p, self.scale, self.offset) for p in poses])
        if self.opt.smooth_path:
            self.poses = smooth_camera_path(self.poses, self.opt.smooth_path_window)
        pose0 = np.array(transform['base_transform_matrix'], dtype=np.float32).reshape(4, 4)
        self.pose0 = nerf_matrix_to_ngp(pose0, self.scale, self.offset)
        lmks = np.array([f['landmarks'] for f in frames], dtype=np.float32)
        lmk0 = np.array(transform['base_landmark'], dtype=np.float32)
        self.lmks = lmks.transpose(0, 2, 1) * self.scale + self.offset
        self.lmk0 = lmk0.T * self.scale + self.offset
        aud_features = np.load(os.path.join(self.root_path, 'aud_af.npy'))
        fids = np.array([int(re.sub(r'\D', '', f['file_path'])) for f in frames])
        self.aud_features = aud_features[fids]
        self.a0 = self.aud_features[0]

        if self.training:
            lm2ds = np.load(os.path.join(self.root_path, 'lms.pkl'), allow_pickle=True)  # [N, 68, 2]

            self.images = []
            self.masks = []
            self.face_rect = []
            self.lips_rect = []
            self.eye_masks = []
            winks = {"left_wink": [], "right_wink": []}
            for f in tqdm.tqdm(frames, desc=f'Loading {mode} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                assert os.path.exists(f_path), f'[ERROR] {f_path} NOT FOUND!'

                if self.preload > 0:
                    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image.astype(np.float32) / 255  # [H, W, 3/4]
                    mask = cv2.imread(f_path.replace("frame_", "parse_"), cv2.IMREAD_UNCHANGED)
                    mask = (mask == [255, 0, 0]).prod(-1).astype(bool)[..., None]
                    image = image * mask + self.bg_img * ~mask
                    self.images.append(image)
                    self.masks.append(mask)
                else:
                    self.images.append(f_path)
                # load aud, lms and extract face

                lms = lm2ds[str(Path(f_path).resolve())]  # [68, 2]
                eyebrows_left = [17, 18, 19, 20, 21, 39, 40, 41, 36]
                eyebrows_right = [22, 23, 24, 25, 26, 45, 46, 47, 42]
                eyes_left = slice(36, 42)
                eyes_right = slice(42, 48)
                area_left = polygon_area(*lms[eyes_left].T) / polygon_area(*lms[eyebrows_left].T)
                area_right = polygon_area(*lms[eyes_right].T) / polygon_area(*lms[eyebrows_right].T)
                winks["left_wink"].append(area_left)
                winks["right_wink"].append(area_right)
                left_eye_mask = cv2.fillConvexPoly(np.zeros((self.H, self.W)), lms[eyebrows_left].astype(int), 1)
                right_eye_mask = cv2.fillConvexPoly(np.zeros((self.H, self.W)), lms[eyebrows_right].astype(int), 1)
                left_eye_mask = left_eye_mask != 0
                right_eye_mask = right_eye_mask != 0
                self.eye_masks.append([left_eye_mask, right_eye_mask])

                xmin, xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max())
                ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
                self.face_rect.append([xmin, xmax, ymin, ymax])

                if self.opt.finetune_lips:
                    xmin, xmax = int(lms[48:60, 1].min()), int(lms[48:60, 1].max())
                    ymin, ymax = int(lms[48:60, 0].min()), int(lms[48:60, 0].max())
                    # padding to H == W
                    cx = (xmin + xmax) // 2
                    cy = (ymin + ymax) // 2
                    l = max(xmax - xmin, ymax - ymin) // 2
                    xmin = max(0, cx - l)
                    xmax = min(self.H, cx + l)
                    ymin = max(0, cy - l)
                    ymax = min(self.W, cy + l)
                    self.lips_rect.append([xmin, xmax, ymin, ymax])

        self.aabb = torch.from_numpy(self.aabb).float()  # [6]
        self.poses = torch.from_numpy(self.poses).float()  # [N, 4, 4]
        self.pose0 = torch.from_numpy(self.pose0).float()  # [4, 4]
        self.R0 = self.pose0[:3, :3]  # [3, 3]
        self.cams = torch.cat([K.expand(self.poses.shape[0], 3, 3), self.poses[:, :3]], -1)
        self.lmks = torch.from_numpy(self.lmks).float()  # [N, 68, 3]
        self.lmk0 = torch.from_numpy(self.lmk0).float()  # [68, 3]
        self.aud_features = torch.from_numpy(self.aud_features).float()  # [N, 64]
        self.a0 = torch.from_numpy(self.a0).float()

        self.bg_img = torch.from_numpy(self.bg_img)
        if self.training:
            if self.preload > 0:
                self.images = torch.from_numpy(np.stack(self.images, axis=0))  # [N, H, W, C]
                self.masks = torch.from_numpy(np.stack(self.masks, axis=0))  # [N, H, W, 1]
            else:
                self.images = np.array(self.images)
            self.left_winks = torch.tensor(winks["left_wink"])
            self.right_winks = torch.tensor(winks["right_wink"])
            if self.opt.smooth_eye:
                ori_left_wink = self.left_winks.clone()
                ori_right_wink = self.right_winks.clone()
                for i in range(ori_left_wink.shape[0]):
                    start = max(0, i - 1)
                    end = min(ori_left_wink.shape[0], i + 2)
                    self.left_winks[i] = ori_left_wink[start:end].mean()
                    self.right_winks[i] = ori_right_wink[start:end].mean()

        if self.preload > 1:
            self.poses = self.poses.to(self.device)
            self.R0 = self.R0.to(self.device)
            self.a0 = self.a0.to(self.device)
            self.cams = self.cams.to(self.device)
            self.lmks = self.lmks.to(self.device)
            self.lmk0 = self.lmk0.to(self.device)
            self.bg_img = self.bg_img.to(torch.half).to(self.device)

            if self.training:
                self.images = self.images.to(torch.half).to(self.device)
                self.masks = self.masks.to(torch.half).to(self.device)
                self.left_winks = self.left_winks.to(self.device)
                self.right_winks = self.right_winks.to(self.device)

    def mirror_index(self, index):
        size = self.poses.shape[0]
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def collate(self, index):
        B = len(index)  # a list of length 1
        # assert B == 1

        results = {}

        # head pose and bg image may mirror (replay --> <-- --> <--).
        index[0] = self.mirror_index(index[0])

        cams = self.cams[index].to(self.device)  # [B, 3, 7]
        results['cams'] = cams
        bg_img = self.bg_img.reshape(1, -1, 3).expand(B, -1, -1).to(self.device)  # [B, N, 3]

        if self.training and self.opt.finetune_lips:
            rect = self.lips_rect[index[0]]
            results['rect'] = rect
            rays = get_rays(cams, self.H, self.W, -1, rect=rect)
        elif self.training and self.opt.finetune_eyes:
            le_mask, re_mask = self.eye_masks[index[0]]
            rays = get_rays(cams, self.H, self.W, -1, mask=torch.from_numpy(le_mask | re_mask).to(self.device))
        elif self.training and self.mode != 'val':
            rays = get_rays(cams, self.H, self.W, self.num_rays, self.opt.patch_size)
        elif self.opt.finetune_eyes and self.mode == 'val':
            le_mask, re_mask = self.eye_masks[index[0]]
            rays = get_rays(cams, self.H, self.W, -1, mask=torch.from_numpy(le_mask | re_mask).to(self.device))
        else:
            rays = get_rays(cams, self.H, self.W)

        results['index'] = index  # for ind. code
        results['H'] = self.H
        results['W'] = self.W
        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']

        results['lmk'] = self.lmks[index].to(self.device)
        results['R'] = cams[..., 3:6]
        results['aud'] = self.aud_features[index].to(self.device)

        # get a mask for rays inside rect_face
        if self.training:
            xmin, xmax, ymin, ymax = self.face_rect[index[0]]
            face_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax)  # [B, N]
            results['face_mask'] = face_mask

        if self.training and self.opt.finetune_eyes:
            results['l_wink'] = self.left_winks[index].to(self.device)
            results['r_wink'] = self.right_winks[index].to(self.device)

            le_mask = torch.from_numpy(le_mask).to(self.device)
            re_mask = torch.from_numpy(re_mask).to(self.device)
            
            results['le_mask'] = torch.gather(le_mask.view(B, -1), 1, rays['inds'])  # [B,N]
            results['re_mask'] = torch.gather(re_mask.view(B, -1), 1, rays['inds'])

        if self.training:
            bg_img = torch.gather(bg_img, 1, torch.stack(3 * [rays['inds']], -1))  # [B, N, 3]

        results['bg_color'] = bg_img

        images = self.images[index]  # [B, H, W, 3/4]
        if self.preload == 0:
            mask = cv2.imread(images[0].replace("frame_", "parse_"), cv2.IMREAD_UNCHANGED)
            mask = (mask == [255, 0, 0]).prod(-1).astype(bool)[..., None]
            mask = torch.from_numpy(mask)
            images = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)  # [H, W, 3]
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            images = images.astype(np.float32) / 255  # [H, W, 3]
            images = torch.from_numpy(images)
            images = images * mask + self.bg_img * ~mask
            images = images.unsqueeze(0)
            mask = mask.unsqueeze(0)
        else:
            mask = self.masks[index]  # [B, H, W, 1]

        images = images.to(self.device)
        mask = mask.to(self.device)

        if self.training:
            C = images.shape[-1]
            images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
            mask = torch.gather(mask.view(B, -1), 1, rays['inds'])  # [B, N]

        results['images'] = images
        results['alphas'] = mask

        return results

    def dataloader(self):
        size = self.poses.shape[0]

        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need poses in trainer.

        return loader
