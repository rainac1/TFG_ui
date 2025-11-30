# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import argparse
import os
import random
import shutil
import pickle
import PIL.Image
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch
import torch.backends.cudnn as cudnn
import trimesh
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from skimage.io import imread
from tqdm import tqdm

from configs.micaconfig import get_cfg_defaults
from datasets.creation.util import get_arcface_input, get_center, draw_on
from utils import micautil
from utils.landmark_detector import LandmarksDetector, detectors
from utils.face_detector import FaceDetector


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False


def process(args, app, image_size=224, draw_bbox=False):
    dst = Path(args.a)
    dst.mkdir(parents=True, exist_ok=True)
    processes = []
    image_paths = sorted(glob(args.i + '/*.png') + glob(args.i + '/*.jpg'))
    fa_mediapipe = FaceDetector('google')
    lm_dense_dict = {}
    for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        name = Path(image_path).stem
        img = cv2.imread(image_path)
        dense_lmk = fa_mediapipe.dense(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        lm_dense_dict[str(image_path)] = dense_lmk[:, :2]
        if idx > 10 and idx % 200 != 0:
            continue
        bboxes, kpss = app.detect(img)
        if bboxes.shape[0] == 0:
            logger.error(f'[ERROR] Face not detected for {image_path}')
            continue
        i = get_center(bboxes, img)
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        blob, aimg = get_arcface_input(face, img)
        file = str(Path(dst, name))
        np.save(file, blob)
        processes.append(file + '.npy')
        cv2.imwrite(file + '.jpg', face_align.norm_crop(img, landmark=face.kps, image_size=image_size))
        if draw_bbox:
            dimg = draw_on(img, [face])
            cv2.imwrite(file + '_bbox.jpg', dimg)

    with open(args.o + '/lms_dense.pkl', 'wb') as fn:
        pickle.dump(lm_dense_dict, fn)
    return processes


def to_batch(path):
    src = path.replace('npy', 'jpg')
    if not os.path.exists(src):
        src = path.replace('npy', 'png')

    image = imread(src)[:, :, :3]
    image = image / 255.
    image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
    image = torch.tensor(image).cuda()[None]

    arcface = np.load(path)
    arcface = torch.tensor(arcface).cuda()[None]

    return image, arcface


def load_checkpoint(args, mica):
    checkpoint = torch.load(args.m)
    if 'arcface' in checkpoint:
        mica.arcface.load_state_dict(checkpoint['arcface'])
    if 'flameModel' in checkpoint:
        mica.flameModel.load_state_dict(checkpoint['flameModel'])


def main(cfg, args):
    device = 'cuda:0'
    cfg.model.testing = True
    mica = micautil.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, device)
    load_checkpoint(args, mica)
    mica.eval()

    faces = mica.flameModel.generator.faces_tensor.cpu()
    Path(args.o).mkdir(exist_ok=True, parents=True)

    app = LandmarksDetector(model=detectors.RETINAFACE)

    with torch.no_grad():
        logger.info(f'Processing has started...')
        paths = process(args, app, draw_bbox=False)
        identitys = []
        for path in tqdm(paths):
            name = Path(path).stem
            images, arcface = to_batch(path)
            codedict = mica.encode(images, arcface)
            opdict = mica.decode(codedict)
            meshes = opdict['pred_canonical_shape_vertices']
            code = opdict['pred_shape_code']
            lmk = mica.flame.compute_landmarks(meshes)

            mesh = meshes[0]
            landmark_51 = lmk[0, 17:]
            landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

            dst = Path(args.a, name)
            dst.mkdir(parents=True, exist_ok=True)
            trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.ply')  # save in millimeters
            trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
            np.save(f'{dst}/identity', code[0].cpu().numpy())
            np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)
            np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)
            identitys.append(code[0].cpu().numpy())

        logger.info(f'Processing finished. Results has been saved in {args.a}')
        np.save(f'{args.o}/identity', np.mean(identitys, 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MICA - Towards Metrical Reconstruction of Human Faces')
    parser.add_argument('-i', default='', type=str, required=True, help='Input folder with images or video filename')
    parser.add_argument('-o', default='', type=str, help='Output folder')
    parser.add_argument('-a', default='', type=str, help='Processed images for MICA input')
    parser.add_argument('-m', default='../../data/pretrained/mica.tar', type=str, help='Pretrained model path')

    args = parser.parse_args()
    args.o = args.i + '/..'
    args.a = args.i + '/../mica_arcface'
    cfg = get_cfg_defaults()

    deterministic(42)
    main(cfg, args)
