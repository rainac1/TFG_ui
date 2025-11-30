from glob import glob
from pathlib import Path

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from loguru import logger
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, config):
        source = Path(config.workspace)
        self.images = []
        self.lm_dict = {}
        self.lm_dense_dict = {}
        self.device = 'cuda:0'
        self.source = source
        self.config = config
        self.initialize()

    def initialize(self):
        path = Path(self.source, 'frame_imgs').resolve()
        self.images = sorted(path.glob('*.[j|p][p|n]g'))
        self.lm_dict = np.load(f'{self.source}/lms.pkl', allow_pickle=True)
        self.lm_dense_dict = np.load(f'{self.source}/lms_dense.pkl', allow_pickle=True)

        if self.config.end_frames == 0:
            self.images = self.images[self.config.begin_frames:]

        elif self.config.end_frames != 0:
            self.images = self.images[self.config.begin_frames:-self.config.end_frames]

        logger.info(f'[ImagesDataset] Initialized with {len(self.images)} frames...')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imagepath = str(self.images[index])
        pil_image = Image.open(imagepath).convert("RGB")
        image = F.to_tensor(pil_image)

        shape_path = Path(self.source, 'identity.npy')
        if shape_path.exists():
            shape = np.load(shape_path)
        else:
            logger.error('[ImagesDataset] Shape (identity.npy) not found! Run MICA shape predictor from https://github.com/Zielon/MICA')
            exit(-1)

        lmk = self.lm_dict[imagepath]
        dense_lmk = self.lm_dense_dict[imagepath]

        lmks = torch.from_numpy(lmk).float()
        dense_lmks = torch.from_numpy(dense_lmk).float()
        shapes = torch.from_numpy(shape).float()

        payload = {
            'image': image,
            'lmk': lmks,
            'dense_lmk': dense_lmks,
            'shape': shapes
        }

        return payload
