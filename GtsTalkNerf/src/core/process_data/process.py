# coding: utf-8
import argparse
import cv2
import gc
import numpy as np
import os
import pickle
import re
import shutil
from multiprocessing import Process, set_start_method, Queue, freeze_support
from pathlib import Path
from scipy.ndimage import binary_erosion, binary_dilation
from tqdm import tqdm, trange


def myprint(s: str):
    col = os.get_terminal_size().columns
    print(s.center(col))


def str2bool(x: str):
    return x.lower() in ['t', '1', 'true', 'y', 'yes']


def polygon_area(x, y):
    return np.abs(sum(x * (np.roll(y, -1) - np.roll(y, 1)))) / 2


def worker(q: Queue, output: Queue):
    """for multiprocessing saving results"""
    for func, args in iter(q.get, 'STOP'):
        out = func(*args)
        output.put(out)


def export_torso(frame_paths, parse_paths, torso_folder, bg_color):
    torso_paths = []
    for frame_path, parse_path in zip(frame_paths, parse_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        parse = cv2.imread(str(parse_path))
        head_part = np.prod(parse == [255, 0, 0], axis=-1, dtype=bool)
        neck_part = np.prod(parse == [0, 255, 0], axis=-1, dtype=bool)
        torso_part = np.prod(parse == [0, 0, 255], axis=-1, dtype=bool)
        bg_part = np.prod(parse == [255, 255, 255], axis=-1, dtype=bool)
        frame[bg_part] = bg_color[bg_part]
        torso_image = frame.copy()
        torso_image[head_part] = bg_color[head_part]
        torso_alpha = 255 * np.ones_like(torso_image[..., :1], dtype=np.uint8)
        # torso part "vertical" in-painting...
        L = 8 + 1
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1)  # [M, 2]
        # lexsort: sort 2D coords first by y then by x,
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid]  # [m, 2]
        # only keep top-is-head pixels
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_torso_coords_up.T)]
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            # get the color
            top_torso_colors = frame[tuple(top_torso_coords.T)]  # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0)  # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None]  # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2)  # [Lm, 2]
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0)  # [L, m, 3]
            darken_scaler = 0.98 ** np.arange(L)[:, None, None]  # [L, 1, 1]
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3)  # [Lm, 3]
            # set color
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

            inpaint_torso_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None

        # neck part "vertical" in-painting...
        push_down = 4
        L = 48 + push_down + 1

        neck_part = binary_dilation(neck_part, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
                                    iterations=3)

        neck_coords = np.stack(np.nonzero(neck_part), axis=-1)  # [M, 2]
        # lexsort: sort 2D coords first by y then by x,
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid]  # [m, 2]
        # only keep top-is-head pixels
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)]

        top_neck_coords = top_neck_coords[mask]
        # push these top down for 4 pixels to make the neck inpainting more natural...
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
        # get the color
        top_neck_colors = frame[tuple(top_neck_coords.T)]  # [m, 3]
        # construct inpaint coords (vertically up, or minus in x)
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0)  # [L, m, 2]
        inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None]  # [L, 1, 2]
        inpaint_neck_coords += inpaint_offsets
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2)  # [Lm, 2]
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0)  # [L, m, 3]
        darken_scaler = 0.98 ** np.arange(L)[:, None, None]  # [L, 1, 1]
        inpaint_neck_colors = (inpaint_neck_colors * darken_scaler).reshape(-1, 3)  # [Lm, 3]
        # set color
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        # apply blurring to the inpaint area to avoid vertical-line artifects...
        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True

        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)

        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # set mask
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0
        torso_paths.append(torso_folder / frame_path.name)
        cv2.imwrite(str(torso_paths[-1]), np.concatenate([torso_image, torso_alpha], axis=-1))
    return torso_paths


def main(args):
    freeze_support()
    set_start_method('spawn')
    inputfn = Path(args.inputfn)
    save_folder = Path(__file__, '../../results', inputfn.stem).resolve()
    frame_folder = save_folder / 'frame_imgs'
    parse_folder = save_folder / 'parse_imgs'
    torso_folder = save_folder / 'torso_imgs'
    steps = list(map(int, re.findall('\\d+', args.steps)))
    save_folder.mkdir(parents=True, exist_ok=True)
    if inputfn.is_file():
        try:
            shutil.copyfile(inputfn, save_folder / inputfn.name)
            inputfn = save_folder / inputfn.name
        except shutil.SameFileError:
            pass
    else:
        inputfn = None
        for fn in save_folder.glob(f"{save_folder.stem}.*"):
            if fn.suffix in ['.avi', '.flv', '.mkv', '.mp4', '.mpeg', '.mov', '.ts']:
                inputfn = fn
                break
        if inputfn is None:
            raise FileNotFoundError(f"No video file found in {save_folder}.")
    if os.name != "nt":
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
    if 1 in steps:
        myprint('Step 1: Export Voice and Images')
        os.system(f"python process_video.py -f {inputfn} -t {args.tar_size}")

    frame_paths = sorted(frame_folder.glob("*.[j|p][p|n]g"))
    assert frame_paths, "Please run step 1 first."
    h, w, _ = cv2.imread(str(frame_paths[0])).shape

    if 2 in steps:
        myprint('Step 2: Face Parsing')
        os.system(f'python face_parsing/test.py --respath={parse_folder} --imgpath={frame_folder}')

    parse_paths = sorted(parse_folder.glob("*.[j|p][p|n]g"))

    if 3 in steps:
        myprint('Step 3: Face Tracking')
        os.chdir('facetracker')
        if not Path(save_folder, 'identity.npy').exists() and not args.regen:
            os.system(f'python run_mica.py -i {frame_folder}')
        os.system(f'python run_tracker.py -i {frame_folder}')
        os.chdir('..')

    if 4 in steps:
        myprint('Step 4: Export Torso Part')
        torso_folder.mkdir(parents=True, exist_ok=True)
        total_num = len(frame_paths)
        chunk = 4 * args.nworkers
        beginids = np.arange(0, total_num, chunk)
        tasks = Queue()
        results = Queue()
        torso_paths = []
        bg_color = cv2.imread(str(save_folder / 'bg.png'))
        for i in beginids:
            tasks.put((export_torso, (frame_paths[i:i+chunk], parse_paths[i:i+chunk], torso_folder, bg_color)))

        for _ in range(args.nworkers):
            tasks.put('STOP')
            Process(target=worker, args=(tasks, results)).start()

        for _ in tqdm(beginids):
            out = results.get()
            torso_paths.extend(out)

        assert results.empty() and len(torso_paths) == total_num, "Multiprocess Error"
        tasks.close()
        results.close()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--inputfn', type=str, required=True)
    parser.add_argument('-s', '--steps', type=str, default='1,2,3')
    parser.add_argument('-t', '--tar_size', type=int, default=512)
    parser.add_argument('-r', '--regen', action='store_true')
    parser.add_argument('-n', '--nworkers', type=int, default=8)

    args = parser.parse_args()
    main(args)
