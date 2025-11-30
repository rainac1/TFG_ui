import argparse
import face_alignment
import imageio.v2 as imageio
import numpy as np
import onnxruntime as ort
import os
import pickle
import re
import shutil
import torch
from collections import OrderedDict
from multiprocessing import Process, set_start_method, Queue, freeze_support
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from tqdm import tqdm


def myprint(s: str):
    col = os.get_terminal_size().columns
    print(s.center(col))


def bbox_from_lms(lms):
    if lms is None:
        return None
    bbox = []
    for lm in lms:
        box = [min(lm[:, 0]), min(lm[:, 1]), max(lm[:, 0]), max(lm[:, 1])]
        box[0] = lm[27, 0] * 2 - box[2]
        box[1] = (lm[27, 1] - 0.38 * box[3]) / 0.62
        bbox.append(box)
    return bbox


def pad_bbox(bbox, img_wh, padding_ratio=0.25):
    x1, y1, x2, y2 = bbox[:4]
    w, h = img_wh
    width = x2 - x1
    height = y2 - y1
    size_bb = int(max(width, height) * (1 + padding_ratio))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(w - x1, size_bb)
    size_bb = min(h - y1, size_bb)
    return [x1, y1, x1 + size_bb, y1 + size_bb]


def worker(q: Queue, output: Queue):
    """for multiprocessing saving results"""
    for func, args in iter(q.get, 'STOP'):
        out = func(*args)
        output.put(out)


def cal_nn(i, masks_path, all_xys):
    fg_xys = np.stack(np.nonzero(imageio.imread(masks_path))).T
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(fg_xys)
    dists, _ = nbrs.kneighbors(all_xys)
    return i, dists


class Segmenter:
    def __init__(self, args):
        sess_opts = ort.SessionOptions()
        providers = ['CPUExecutionProvider']
        if args.device.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            self.device = args.device
        else:
            self.device = 'cpu'
        if not args.disable_openmp:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            if not os.environ.get('OMP_NUM_THREADS', 0):
                os.environ['OMP_NUM_THREADS'] = '8'
            sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
            sess_opts.intra_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

        self.session = ort.InferenceSession(args.onnx_model_path, sess_opts, providers=providers)

        self.means = np.array([[[0.485, 0.456, 0.406]]])
        self.stds = np.array([[[0.229, 0.224, 0.225]]])
        if isinstance(args.inp_size, str):
            args.inp_size = map(int, re.findall(r'\d+', args.inp_size))
        self.inp_size = tuple(args.inp_size)
        assert len(self.inp_size) == 2
        self.out_size = (args.tar_size, args.tar_size)

        self.inp_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        self.binding = self.session.io_binding()

    def run(self, image: Image.Image) -> Image.Image:
        img = image.resize(self.inp_size, resample=Image.LANCZOS)
        img = ((np.array(img) / 255 - self.means) / self.stds).astype(np.float32)
        img = np.moveaxis(img, -1, 0)[None]
        self.binding.bind_cpu_input(self.inp_name, img)
        self.binding.bind_output(self.out_name, self.device)
        self.session.run_with_iobinding(self.binding)
        out = self.binding.get_outputs()[0].numpy().squeeze()
        out = np.round((out - out.min()) / (out.max() - out.min()) * 255).astype(np.uint8)
        out = (out > 127).astype(np.uint8) * 255
        out = Image.fromarray(out, 'L')
        return out.resize(self.out_size, resample=Image.LANCZOS)


def main(args):
    inputfn = Path(args.inputfn)
    save_folder = inputfn.parent
    frame_folder = save_folder / 'frame_imgs'
    masks_folder = save_folder / 'masks_imgs'
    face_videofn = save_folder / (save_folder.stem + '_face' + inputfn.suffix)
    if frame_folder.exists():
        shutil.rmtree(frame_folder)
    if masks_folder.exists():
        shutil.rmtree(masks_folder)
    frame_folder.mkdir(parents=True, exist_ok=True)
    masks_folder.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(inputfn)
    fps = reader.get_meta_data()['fps']
    w, h = reader.get_meta_data()['size']

    if fps != 25:
        myprint("The video should be 25 fps, Processing")
        inputfn_25 = save_folder / (save_folder.stem + '_25fps' + inputfn.suffix)
        os.system(f'ffmpeg -i {inputfn} -qscale 0 -r 25 -y {inputfn_25} -loglevel warning')
        inputfn = inputfn_25
        reader = imageio.get_reader(inputfn)
        fps = reader.get_meta_data()['fps']
        assert fps == 25

    myprint("Exporting Voice and Box")
    wav_file = save_folder / (save_folder.stem + '.wav')
    os.system(f'ffmpeg -i {inputfn} -f wav -ar 16000 -y {wav_file} -loglevel warning')

    if hasattr(face_alignment.LandmarksType, 'TWO_D'):
        landmarks_type = face_alignment.LandmarksType.TWO_D
    else:
        landmarks_type = face_alignment.LandmarksType._2D

    fa = face_alignment.FaceAlignment(landmarks_type, device=args.device)
    xmin, ymin, xmax, ymax = 1e5, 1e5, -1, -1
    for i, frame in tqdm(enumerate(reader), total=reader.count_frames()):
        if not i % 50 == 0:
            continue
        bboxes = fa.face_detector.detect_from_image(frame)
        assert len(bboxes) > 0, "No face"
        if len(bboxes) > 1 and i == 0:
            print("More than one face detected, First one is chosen")
        x1, y1, x2, y2, _ = bboxes[0]
        xmin = min(x1, xmin)
        ymin = min(y1, ymin)
        xmax = max(x2, xmax)
        ymax = max(y2, ymax)
    x1, y1, x2, y2 = pad_bbox([xmin, ymin, xmax, ymax], [w, h])
    with open(save_folder / 'bbox.txt', 'w') as fp:
        fp.write(f"xmin:{x1}\nymin:{y1}\nxmax:{x2}\nymax:{y2}\nw:{w}\nh:{h}")
    myprint("Saving Frame Images and Detecting Landmarks")
    segmenter = Segmenter(args)
    face_video = imageio.get_writer(face_videofn, fps=25)
    lm_dict = OrderedDict()

    pre_lms = None
    frame_paths = []
    masks_paths = []
    fa = face_alignment.FaceAlignment(landmarks_type, device=args.device, flip_input=True)
    for i, frame in tqdm(enumerate(reader), total=reader.count_frames()):
        img = Image.fromarray(frame[y1:y2, x1:x2], mode='RGB')
        msk = segmenter.run(img)
        img = img.resize(msk.size, resample=Image.LANCZOS)
        img_fn = frame_folder / f'{i:04d}.png'
        msk_fn = masks_folder / f'{i:04d}.png'

        img.save(img_fn)
        msk.save(msk_fn)
        frame_paths.append(img_fn)
        masks_paths.append(msk_fn)

        img = np.array(img)
        face_video.append_data(img)
        lms_lst = fa.get_landmarks_from_image(img, detected_faces=bbox_from_lms(pre_lms))
        pre_lms = lms_lst
        lm_dict[str(img_fn)] = lms_lst[0][:, :2]

    face_video.close()
    with open(save_folder / 'lms.pkl', 'wb') as fn:
        pickle.dump(lm_dict, fn)

    myprint("Exporting Background Image")
    total_num = reader.count_frames()
    sel_ids = np.arange(0, total_num - 5, 25)
    sel_ids += np.random.randint(low=0, high=5, size=len(sel_ids))
    all_xys = np.mgrid[0:args.tar_size, 0:args.tar_size].reshape((2, -1)).transpose()
    tasks = Queue()
    results = Queue()
    imgs = []
    distss = []

    for i in sel_ids:
        tasks.put((cal_nn, (i, masks_paths[i], all_xys.copy())))

    for _ in range(args.nworkers):
        tasks.put('STOP')
        Process(target=worker, args=(tasks, results)).start()

    for _ in tqdm(sel_ids):
        out = results.get()
        imgs.append(imageio.imread(frame_paths[out[0]]).reshape((-1, 3)))
        distss.append(out[1])

    assert results.empty(), "Multiprocess Error"
    tasks.close()
    results.close()

    imgs = np.stack(imgs, 0)
    distss = np.stack(distss, 0)
    # print(distss.shape)

    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)
    bg_pixs = max_dist > 5
    bg_pixs_id = np.nonzero(bg_pixs)
    bg_ids = max_id[bg_pixs]

    bg_img = np.zeros_like(imgs[0])
    bg_img[bg_pixs_id, :] = imgs[bg_ids, bg_pixs_id, :]
    bg_img = bg_img.reshape((args.tar_size, args.tar_size, -1))
    bg_pixs = max_dist.reshape((args.tar_size, args.tar_size)) > 5
    bg_xys = np.stack(np.nonzero(~bg_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bg_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    # print(fg_xys.shape)
    # print(np.max(bg_fg_xys), np.min(bg_fg_xys))
    bg_img[bg_xys[:, 0], bg_xys[:, 1], :] = bg_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
    imageio.imwrite(save_folder / 'bg.png', bg_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--inputfn', type=str, required=True)
    parser.add_argument('-t', '--tar_size', type=int, default=512)
    parser.add_argument('-m', '--onnx_model_path', type=str, default='')
    parser.add_argument('-i', '--inp_size', type=str, default='')
    parser.add_argument('-n', '--nworkers', type=int, default=8)
    parser.add_argument('--disable_openmp', action='store_true')

    args = parser.parse_args()
    if args.onnx_model_path and Path(args.onnx_model_path).exists():
        args.onnx_model_path = Path(args.onnx_model_path)
    else:
        args.onnx_model_path = Path(__file__, '../../data/pretrained/u2net_human_seg.onnx').resolve()
        args.inp_size = '320,320'
        if not args.onnx_model_path.exists():
            url = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx'
            torch.hub.download_url_to_file(url, str(args.onnx_model_path))

    args.device = 'cuda' if torch.cuda.is_available() else "cpu"

    main(args)
